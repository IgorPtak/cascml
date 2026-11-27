import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from cascml.cli import main
from cascml.data import write_samples
from cascml.export import ExportError, check_contract, export_species, open_session
from cascml.train import train_species
from cascml.validate import read_scores, scores_path, write_scores

# Early stopping really triggers, so export must also handle a booster that was trimmed after training.
PARAMS = {
    "n_estimators": 200, "max_depth": 2, "learning_rate": 1.0, "eval_metric": "auc",
    "early_stopping_rounds": 2, "n_jobs": 1,
}


@pytest.fixture
def trained(cfg, tmp_path, make_sample):
    """Species dir with a trained model 'm'; one candidate in ten has a sentinel, as O2 writes them."""
    df = make_sample(cfg, signal_shift=1.0)
    df.loc[df.index % 10 == 0, "cascradius"] = -999.0
    sdir = write_samples(df, cfg, "xi", [Path("/data/fake.root")], tmp_path).parent
    train_species(sdir, PARAMS, "m", seed=42)
    return sdir


def test_export_matches_validated_scores_including_sentinels(trained):
    records = export_species(trained, "m", "onnx", opset=15, seed=0)
    assert len(records) == 3
    for r in records:
        path = trained / "models" / "m" / "onnx" / r["file"]
        assert r["file"].startswith("xi_pt_")
        assert r["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert r["max_abs_diff"] <= 1e-5
        assert r["n_parity_sentinel"] > 0
        assert r["us_per_candidate_batch1"] > 0


def test_export_rejects_other_sample_hash(trained):
    meta_path = trained / "models" / "m" / "train_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["sample_hash"] = "0000"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ExportError, match="sample_hash mismatch"):
        export_species(trained, "m", "onnx", opset=15, seed=0)


def test_export_aborts_when_onnx_disagrees_with_scores(trained):
    # Scores no longer come from this model (e.g. model.ubj overwritten): the exported file must not be trusted.
    path = scores_path(trained, "m")
    scores, meta = read_scores(path)
    write_scores(path, scores["cand_id"], scores["score"] + 1e-3, meta["sample_hash"], meta["model"])
    with pytest.raises(ExportError, match="differs from scores.parquet"):
        export_species(trained, "m", "onnx", opset=15, seed=0)


def test_export_without_training(trained):
    with pytest.raises(ExportError, match="run cascml train"):
        export_species(trained, "other", "onnx", opset=15, seed=0)


def test_contract_rejects_wrong_number_of_features(trained):
    export_species(trained, "m", "onnx", opset=15, seed=0)
    session = open_session(next((trained / "models" / "m" / "onnx").glob("*.onnx")))
    check_contract(session, 12)
    with pytest.raises(ExportError, match=r"float input \[None, 13\]"):
        check_contract(session, 13)


def test_main_export(trained, cfg, tmp_path, capsys):
    config = tmp_path / "analysis.yml"
    config.write_text(yaml.safe_dump(cfg))
    assert main(["export", "-c", str(config), "--output", str(tmp_path), "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert out.count("parity max |diff|") == 3
