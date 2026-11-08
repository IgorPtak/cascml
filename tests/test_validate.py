import numpy as np
import pandas as pd
import pytest

from cascml.validate import (
    ValidationError,
    attach_scores,
    auc_table,
    bootstrap_auc,
    read_scores,
    roc_auc,
    scores_path,
    write_scores
)

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


def test_roc_auc_hand_computed():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


@pytest.mark.parametrize("score, expected", [
    ([0.1, 0.2, 0.8, 0.9], 1.0),
    ([0.9, 0.8, 0.2, 0.1], 0.0),
    ([0.5, 0.5, 0.5, 0.5], 0.5),
], ids=["perfect", "inverted", "constant"])
def test_roc_auc_limits(score, expected):
    assert roc_auc([0, 0, 1, 1], score) == pytest.approx(expected)


def test_roc_auc_random_score_is_near_half():
    rng = np.random.default_rng(1)
    label = rng.integers(0, 2, size=20_000)
    assert roc_auc(label, rng.random(20_000)) == pytest.approx(0.5, abs=0.02)


def test_roc_auc_depends_only_on_order():
    label = [0, 1, 0, 1, 1, 0]
    score = np.array([0.2, 0.7, 0.4, 0.9, 0.3, 0.1])
    assert roc_auc(label, score) == pytest.approx(roc_auc(label, np.log(score) * 100 - 7))


def test_roc_auc_rejects_single_class():
    with pytest.raises(ValidationError, match="both classes"):
        roc_auc([1, 1, 1], [0.2, 0.5, 0.9])


def test_auc_table_per_bin():
    df = pd.DataFrame({
        "pt_bin": [0, 0, 0, 1, 1, 1, 1],
        "label":  [0, 1, 1, 0, 0, 1, 1],
        "score":  [0.1, 0.8, 0.9, 0.9, 0.8, 0.2, 0.1],
    })
    table = auc_table(df, by=["pt_bin"])
    assert list(table["pt_bin"]) == [0, 1]
    assert list(table["n_signal"]) == [2, 2]
    assert list(table["n_bkg"]) == [1, 2]
    assert list(table["auc"]) == pytest.approx([1.0, 0.0])


def random_case(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n), rng.random(n)


def test_bootstrap_interval_contains_estimate():
    label, score = random_case(2000, 1)
    auc, lo, hi = bootstrap_auc(label, score + 0.3 * label, n_boot=200, ci=0.95)
    assert lo < auc < hi


def test_bootstrap_coverage_for_random_classifier():
    rng = np.random.default_rng(10)
    n_repeats = 100
    hits = 0
    for r in range(n_repeats):
        label = rng.integers(0, 2, size=200)
        _, lo, hi = bootstrap_auc(label, rng.random(200), n_boot=100, ci=0.95, seed=r)
        hits += lo <= 0.5 <= hi
    assert 0.88 <= hits / n_repeats <= 1.0


def test_bootstrap_interval_shrinks_with_statistics():
    widths = []
    for n in (1000, 4000):
        label, score = random_case(n, 3)
        _, lo, hi = bootstrap_auc(label, score + 0.3 * label, n_boot=200, ci=0.95)
        widths.append(hi - lo)
    assert widths[1] == pytest.approx(widths[0] / 2, rel=0.3)


def test_bootstrap_perfect_classifier_has_no_spread():
    assert bootstrap_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_boot=50, ci=0.95) == (1.0, 1.0, 1.0)


def test_bootstrap_is_reproducible():
    label, score = random_case(500, 4)
    assert bootstrap_auc(label, score, 100, 0.95, seed=7) == bootstrap_auc(label, score, 100, 0.95, seed=7)


def test_auc_table_with_bootstrap_adds_interval():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "pt_bin": np.repeat([0, 1], 500),
        "label": rng.integers(0, 2, size=1000),
        "score": rng.random(1000),
    })
    table = auc_table(df, by=["pt_bin"], n_boot=50, ci=0.95)
    assert list(table.columns) == ["pt_bin", "n_signal", "n_bkg", "auc", "auc_lo", "auc_hi"]
    assert (table["auc_lo"] <= table["auc"]).all()
    assert (table["auc"] <= table["auc_hi"]).all()


def test_auc_table_without_bootstrap_has_no_interval():
    df = pd.DataFrame({"pt_bin": [0, 0, 0, 0], "label": [0, 0, 1, 1], "score": [0.1, 0.4, 0.35, 0.8]})
    assert list(auc_table(df, by=["pt_bin"]).columns) == ["pt_bin", "n_signal", "n_bkg", "auc"]
