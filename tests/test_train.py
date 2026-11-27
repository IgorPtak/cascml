import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
import yaml

from cascml.cli import ConfigError, main, model_params_for
from cascml.data import DataError, bin_dir_name, read_manifest, read_sample, write_samples
from cascml.train import feature_matrix, resolve_params, train_species
from cascml.validate import read_scores, scores_path

# Small and single-threaded: fast and bit-for-bit reproducible.
FAST_PARAMS = {
    "n_estimators": 20,
    "max_depth": 2,
    "learning_rate": 0.3,
    "tree_method": "hist",
    "eval_metric": "auc",
    "early_stopping_rounds": 5,
    "n_jobs": 1,
}


@pytest.fixture
def species_dir(cfg, tmp_path, make_sample):
    df = make_sample(cfg, signal_shift=2.0)
    manifest_path = write_samples(df, cfg, "xi", [Path("/data/fake.root")], tmp_path)
    return manifest_path.parent, df


# resolve_params


def test_resolve_params_auto_is_bkg_over_signal():
    resolved = resolve_params({"scale_pos_weight": "auto", "max_depth": 3}, np.array([0, 0, 0, 1]))
    assert resolved == {"scale_pos_weight": 3.0, "max_depth": 3}


def test_resolve_params_does_not_mutate_shared_params():
    # The same dict is reused for every pT bin; overwriting "auto" would leak bin 0's weight into bin 1.
    params = {"scale_pos_weight": "auto"}
    resolve_params(params, np.array([0, 0, 1]))
    assert params == {"scale_pos_weight": "auto"}


@pytest.mark.parametrize("params", [{"scale_pos_weight": 2.5}, {"max_depth": 3}])
def test_resolve_params_leaves_explicit_or_missing_weight(params):
    assert resolve_params(params, np.array([0, 0, 0, 1])) == params


def test_resolve_params_auto_without_signal():
    with pytest.raises(DataError, match="needs signal"):
        resolve_params({"scale_pos_weight": "auto"}, np.array([0, 0, 0]))


# config validation


@pytest.mark.parametrize("value", ["auto", 2.5, 10])
def test_model_params_accepts_scale_pos_weight(value):
    assert model_params_for({"default": {"scale_pos_weight": value}}, "xi")["scale_pos_weight"] == value


@pytest.mark.parametrize("value", [True, 0, -1.0, "Auto", "balanced"])
def test_model_params_rejects_bad_scale_pos_weight(value):
    with pytest.raises(ConfigError, match="scale_pos_weight"):
        model_params_for({"default": {"scale_pos_weight": value}}, "xi")


def test_model_params_per_species_overrides_default():
    params = {"default": {"max_depth": 4, "n_estimators": 300}, "per_species": {"xi": {"max_depth": 2}}}
    assert model_params_for(params, "xi") == {"max_depth": 2, "n_estimators": 300}
    assert model_params_for(params, "omega") == {"max_depth": 4, "n_estimators": 300}


# train_species


def test_train_species_writes_scores_models_and_meta(species_dir, cfg):
    sdir, df = species_dir
    out_path, stats = train_species(sdir, FAST_PARAMS, "m", seed=42)
    manifest = read_manifest(sdir)

    assert out_path == scores_path(sdir, "m")
    scores, scores_meta = read_scores(out_path)
    assert sorted(scores["cand_id"]) == sorted(df["cand_id"])
    assert scores["score"].between(0.0, 1.0).all()
    assert scores_meta["sample_hash"] == manifest["sample_hash"]

    for entry in manifest["bins"]:
        bin_dir = bin_dir_name(entry["pt_lo"], entry["pt_hi"])
        assert (sdir / "models" / "m" / bin_dir / "model.ubj").is_file()

    meta = json.loads((sdir / "models" / "m" / "train_meta.json").read_text())
    assert meta["model"] == "m"
    assert meta["family"] == "xgboost"
    assert meta["xgboost_version"] == xgb.__version__
    assert meta["case"] == cfg["case"]
    assert meta["species"] == "xi"
    assert meta["sample_hash"] == manifest["sample_hash"]
    assert meta["features"] == manifest["features"]
    assert meta["pt_bins"] == manifest["pt_bins"]
    assert meta["seed"] == 42
    assert meta["params"] == {"random_state": 42, **FAST_PARAMS}
    assert meta["bins"] == json.loads(json.dumps(stats))
    assert len(meta["bins"]) == len(manifest["bins"])
    for row in meta["bins"]:
        assert row["train_time_s"] >= 0.0
        assert row["scale_pos_weight"] is None
        assert row["params_resolved"] == meta["params"]


def test_train_species_auto_weight_uses_only_training_folds(cfg, tmp_path, make_sample):
    df = make_sample(cfg, signal_shift=2.0)
    # Keep one signal in three in the training folds, so that n_bkg / n_sig is clearly not 1.
    in_train_folds = df["fold"] > 0
    drop = in_train_folds & (df["label"] == 1) & (df.index % 3 != 0)
    df = df[~drop].reset_index(drop=True)
    sdir = write_samples(df, cfg, "xi", [Path("/data/fake.root")], tmp_path).parent

    _, stats = train_species(sdir, {**FAST_PARAMS, "scale_pos_weight": "auto"}, "m", seed=42)

    for b, row in enumerate(stats):
        train = df[(df["pt_bin"] == b) & (df["fold"] > 0)]
        expected = (train["label"] == 0).sum() / (train["label"] == 1).sum()
        assert expected > 1.5
        assert row["scale_pos_weight"] == pytest.approx(expected)
        assert row["params_resolved"]["scale_pos_weight"] == pytest.approx(expected)
    meta = json.loads((sdir / "models" / "m" / "train_meta.json").read_text())
    assert meta["params"]["scale_pos_weight"] == "auto"


def test_reloaded_model_reproduces_scores(species_dir):
    # Guards the feature order / float32 contract that ONNX export will rely on.
    sdir, _ = species_dir
    out_path, _ = train_species(sdir, FAST_PARAMS, "m", seed=42)
    scores, _ = read_scores(out_path)
    manifest = read_manifest(sdir)
    for entry in manifest["bins"]:
        sample, _ = read_sample(entry["file"], manifest["sample_hash"])
        clf = xgb.XGBClassifier()
        clf.load_model(sdir / "models" / "m" / bin_dir_name(entry["pt_lo"], entry["pt_hi"]) / "model.ubj")
        reloaded = clf.predict_proba(feature_matrix(sample, manifest["features"]))[:, 1]
        expected = scores.set_index("cand_id").loc[sample["cand_id"], "score"].to_numpy()
        assert np.max(np.abs(reloaded - expected)) == 0.0


def test_early_stopped_model_file_holds_only_used_trees(species_dir):
    # A plain Booster ignores best_iteration; after trimming it must give the same scores as validation.
    sdir, _ = species_dir
    params = {**FAST_PARAMS, "n_estimators": 200, "learning_rate": 1.0, "early_stopping_rounds": 2}
    out_path, stats = train_species(sdir, params, "m", seed=42)
    scores = read_scores(out_path)[0].set_index("cand_id")["score"]
    manifest = read_manifest(sdir)
    for entry, row in zip(manifest["bins"], stats):
        n_used = row["best_iteration"] + 1
        assert row["n_rounds_trained"] > n_used  # early stopping really left extra trees behind

        booster = xgb.Booster()
        booster.load_model(sdir / "models" / "m" / bin_dir_name(entry["pt_lo"], entry["pt_hi"]) / "model.ubj")
        assert booster.num_boosted_rounds() == n_used

        sample, _ = read_sample(entry["file"], manifest["sample_hash"])
        raw = booster.predict(xgb.DMatrix(feature_matrix(sample, manifest["features"])))
        assert np.max(np.abs(raw - scores.loc[sample["cand_id"]].to_numpy())) == 0.0


def test_train_species_is_reproducible(species_dir):
    sdir, _ = species_dir
    first, _ = read_scores(train_species(sdir, FAST_PARAMS, "a", seed=42)[0])
    second, _ = read_scores(train_species(sdir, FAST_PARAMS, "b", seed=42)[0])
    assert first.equals(second)


def test_train_species_without_samples(tmp_path):
    with pytest.raises(DataError, match="run cascml prepare"):
        train_species(tmp_path / "nothing", FAST_PARAMS, "m", seed=42)


# CLI: train -> validate


def test_main_train_then_validate(species_dir, cfg, tmp_path, capsys):
    sdir, _ = species_dir
    cfg["validation"]["bootstrap"]["n"] = 20
    config = tmp_path / "analysis.yml"
    config.write_text(yaml.safe_dump(cfg))
    params = tmp_path / "params.yml"
    params.write_text(yaml.safe_dump({"default": {**FAST_PARAMS, "scale_pos_weight": "auto"}}))

    assert main(["train", "-c", str(config), "-p", str(params), "--output", str(tmp_path), "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert "species xi, model m" in out
    assert "scale_pos_weight=1.0" in out

    assert main(["validate", "-c", str(config), "--output", str(tmp_path), "--model", "m"]) == 0
    metrics = json.loads((sdir / "models" / "m" / "metrics.json").read_text())
    assert all(row["auc"] > 0.9 for row in metrics["auc_test"])


def test_main_train_rejects_bad_params(species_dir, cfg, tmp_path, capsys):
    config = tmp_path / "analysis.yml"
    config.write_text(yaml.safe_dump(cfg))
    params = tmp_path / "params.yml"
    params.write_text(yaml.safe_dump({"default": {**FAST_PARAMS, "scale_pos_weight": "balanced"}}))
    assert main(["train", "-c", str(config), "-p", str(params), "--output", str(tmp_path), "--model", "m"]) == 1
    assert "scale_pos_weight" in capsys.readouterr().err
