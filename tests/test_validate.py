import numpy as np
import pandas as pd
import pytest

from cascml.validate import ValidationError, attach_scores, read_scores, scores_path, write_scores

HASH = "abc123"


@pytest.fixture
def sample():
    # cand_id with gaps, as after selection: candidates outside the mass window were removed.
    return pd.DataFrame({
        "cand_id": np.array([3, 7, 8, 12, 20, 21], dtype=np.int64),
        "label": np.array([1, 0, 1, 0, 1, 0], dtype=np.int8),
        "pt_bin": np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
    })


def good_scores(sample):
    return pd.DataFrame({"cand_id": sample["cand_id"], "score": np.linspace(0.0, 1.0, len(sample))})


def test_scores_roundtrip(tmp_path):
    path = scores_path(tmp_path / "xi", "dummy")
    write_scores(path, [5, 1, 9], [0.1, 0.9, 0.5], sample_hash=HASH, model_name="dummy")
    df, meta = read_scores(path)
    assert path == tmp_path / "xi" / "models" / "dummy" / "scores.parquet"
    assert df["cand_id"].dtype == np.int64
    assert df["score"].dtype == np.float64
    assert list(df["cand_id"]) == [5, 1, 9]
    assert meta == {"sample_hash": HASH, "model": "dummy"}


def test_read_scores_rejects_foreign_parquet(tmp_path):
    path = tmp_path / "other.parquet"
    pd.DataFrame({"cand_id": [1], "score": [0.5]}).to_parquet(path)
    with pytest.raises(ValidationError, match="not a cascml score file"):
        read_scores(path)


def test_attach_scores_joins_by_cand_id_not_by_order(sample):
    scores = pd.DataFrame({"cand_id": sample["cand_id"], "score": sample["label"].astype(float)})
    out = attach_scores(sample, scores.iloc[::-1], {"sample_hash": HASH}, HASH)
    assert list(out["cand_id"]) == list(sample["cand_id"])
    assert (out["score"] == out["label"]).all()


def test_attach_scores_rejects_other_sample_hash(sample):
    with pytest.raises(ValidationError, match="recompute the scores"):
        attach_scores(sample, good_scores(sample), {"sample_hash": "other"}, HASH)


@pytest.mark.parametrize("corrupt, message", [
    (lambda s: s.assign(score=s["score"].where(s.index != 2, np.nan)), "not finite"),
    (lambda s: s.assign(score=s["score"].where(s.index != 2, np.inf)), "not finite"),
    (lambda s: pd.concat([s, s.iloc[:1]]), "duplicated"),
    (lambda s: pd.concat([s, pd.DataFrame({"cand_id": [999], "score": [0.5]})]), "not in the sample"),
    (lambda s: s.iloc[1:], "without a score"),
], ids=["nan", "inf", "duplicate", "unknown", "missing"])
def test_attach_scores_rejects_bad_scores(sample, corrupt, message):
    with pytest.raises(ValidationError, match=message):
        attach_scores(sample, corrupt(good_scores(sample)), {"sample_hash": HASH}, HASH)
