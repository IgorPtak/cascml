import numpy as np
import pandas as pd
import pytest

from cascml.validate import (
    SENTINEL_THRESHOLD,
    ValidationError,
    attach_scores,
    auc_table,
    bootstrap_auc,
    efficiency_table,
    ks_train_test,
    read_scores,
    roc_auc,
    scores_path,
    sentinel_table,
    thresholds_for_efficiency,
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


def ten_signals_ten_bkg():
    return pd.DataFrame({
        "pt_bin": 0,
        "label": [1] * 10 + [0] * 10,
        "score": list(np.arange(1, 11) / 10) + list(np.arange(10) / 20),
    })


def test_threshold_keeps_target_fraction_of_signal():
    t = thresholds_for_efficiency(ten_signals_ten_bkg(), by=["pt_bin"], targets=[0.7, 0.9])
    assert list(t.columns) == ["pt_bin", "target", "threshold"]
    assert list(t["threshold"]) == pytest.approx([0.4, 0.2])


def test_threshold_with_ties_keeps_at_least_target():
    df = pd.DataFrame({"pt_bin": 0, "label": [1, 1, 1, 1, 0], "score": [0.9, 0.5, 0.5, 0.5, 0.1]})
    t = thresholds_for_efficiency(df, by=["pt_bin"], targets=[0.5])
    eff = efficiency_table(df, t, by=["pt_bin"])
    assert eff["eff_signal"].iloc[0] == 1.0


def test_threshold_is_not_spoiled_by_float_rounding():
    # 0.55 * 100 == 55.00000000000001 in floating point
    df = pd.DataFrame({"pt_bin": 0, "label": 1, "score": np.arange(100) / 100})
    t = thresholds_for_efficiency(df, by=["pt_bin"], targets=[0.55])
    assert (df["score"] >= t["threshold"].iloc[0]).sum() == 55


def test_threshold_rejects_group_without_signal():
    df = pd.DataFrame({"pt_bin": [0, 0, 1], "label": [1, 0, 0], "score": [0.9, 0.1, 0.5]})
    with pytest.raises(ValidationError, match="no signal"):
        thresholds_for_efficiency(df, by=["pt_bin"], targets=[0.8])


def test_efficiency_table_hand_computed():
    df = ten_signals_ten_bkg()
    t = thresholds_for_efficiency(df, by=["pt_bin"], targets=[0.7])
    row = efficiency_table(df, t, by=["pt_bin"]).iloc[0]
    assert row["eff_signal"] == pytest.approx(0.7)
    assert row["eff_bkg"] == pytest.approx(0.2)
    assert row["eff_signal_err"] == pytest.approx(np.sqrt(0.7 * 0.3 / 10))
    assert (row["n_signal"], row["n_bkg"]) == (10, 10)


def test_efficiency_table_per_charge_with_thresholds_per_bin():
    df = pd.DataFrame({
        "pt_bin": 0,
        "charge": [-1, -1, -1, -1, 1, 1, 1, 1],
        "label":  [1, 1, 0, 0, 1, 1, 0, 0],
        "score":  [0.9, 0.8, 0.1, 0.2, 0.9, 0.3, 0.1, 0.6],
    })
    t = pd.DataFrame({"pt_bin": [0], "target": [0.8], "threshold": [0.5]})
    eff = efficiency_table(df, t, by=["pt_bin", "charge"])
    assert list(eff["charge"]) == [-1, 1]
    assert list(eff["eff_signal"]) == pytest.approx([1.0, 0.5])
    assert list(eff["eff_bkg"]) == pytest.approx([0.0, 0.5])


def test_efficiency_table_rejects_threshold_keys_outside_grouping():
    df = ten_signals_ten_bkg().assign(charge=1)
    t = thresholds_for_efficiency(df, by=["pt_bin", "charge"], targets=[0.8])
    with pytest.raises(ValidationError, match="subset"):
        efficiency_table(df, t, by=["pt_bin"])


def test_efficiency_table_rejects_group_without_a_class():
    df = pd.DataFrame({"pt_bin": 0, "charge": [-1, -1, 1], "label": [1, 0, 0], "score": [0.9, 0.1, 0.5]})
    t = pd.DataFrame({"pt_bin": [0], "target": [0.8], "threshold": [0.5]})
    with pytest.raises(ValidationError, match="need both"):
        efficiency_table(df, t, by=["pt_bin", "charge"])


def test_threshold_from_validation_holds_on_independent_test():
    rng = np.random.default_rng(3)

    def sample(n):
        label = rng.integers(0, 2, size=n)
        return pd.DataFrame({"pt_bin": 0, "label": label, "score": rng.normal(label, 1.0)})

    t = thresholds_for_efficiency(sample(20_000), by=["pt_bin"], targets=[0.8])
    row = efficiency_table(sample(20_000), t, by=["pt_bin"]).iloc[0]
    assert abs(row["eff_signal"] - 0.8) < 3 * row["eff_signal_err"] + 0.01


def make_folds(rng, n, shift_train=0.0):
    label = rng.integers(0, 2, size=n)
    fold = rng.choice([-1, 0, 1, 2, 3, 4], size=n)
    score = rng.normal(label, 1.0) + shift_train * (fold >= 1)
    return pd.DataFrame({"pt_bin": 0, "label": label, "fold": fold, "score": score})


def test_ks_table_columns_and_counts():
    df = make_folds(np.random.default_rng(0), 3000)
    table = ks_train_test(df, by=["pt_bin"])
    assert list(table.columns) == ["pt_bin", "class", "n_train", "n_test", "ks_stat", "p_value"]
    assert list(table["class"]) == ["signal", "bkg"]
    sig = df[df["label"] == 1]
    assert table["n_train"].iloc[0] == (sig["fold"] >= 1).sum()
    assert table["n_test"].iloc[0] == (sig["fold"] == -1).sum()


def test_ks_ignores_validation_fold():
    df = make_folds(np.random.default_rng(1), 3000)
    df.loc[df["fold"] == 0, "score"] += 100.0
    assert ks_train_test(df, by=["pt_bin"])["ks_stat"].max() < 0.1


def test_ks_detects_overtraining():
    table = ks_train_test(make_folds(np.random.default_rng(2), 20_000, shift_train=0.3), by=["pt_bin"])
    assert (table["p_value"] < 1e-6).all()


def test_ks_p_values_are_uniform_without_overtraining():
    rng = np.random.default_rng(3)
    p = np.concatenate([ks_train_test(make_folds(rng, 600), by=["pt_bin"])["p_value"].to_numpy() for _ in range(200)])
    assert 0.02 <= (p < 0.05).mean() <= 0.09
    assert 0.40 <= (p < 0.5).mean() <= 0.60


def test_ks_rejects_group_without_test_candidates():
    df = make_folds(np.random.default_rng(4), 200)
    df.loc[df["label"] == 1, "fold"] = 1
    with pytest.raises(ValidationError, match="need both"):
        ks_train_test(df, by=["pt_bin"])


def test_sentinel_table_counts_per_group_and_class():
    df = pd.DataFrame({
        "pt_bin": [0, 0, 0, 0, 1, 1],
        "label":  [1, 1, 0, 0, 1, 0],
        "a":      [1000.0, 0.5, 1000.0, 1000.0, 0.1, 0.2],
        "b":      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    table = sentinel_table(df, ["a", "b"], by=["pt_bin"])
    assert list(table.columns) == ["pt_bin", "class", "feature", "n", "n_sentinel", "fraction"]
    a = table[table["feature"] == "a"].set_index(["pt_bin", "class"])
    assert a.loc[(0, "signal"), "n_sentinel"] == 1
    assert a.loc[(0, "signal"), "fraction"] == pytest.approx(0.5)
    assert a.loc[(0, "bkg"), "n_sentinel"] == 2
    assert a.loc[(1, "signal"), "n_sentinel"] == 0
    assert (table.loc[table["feature"] == "b", "n_sentinel"] == 0).all()
    assert len(table) == 2 * 2 * 2


@pytest.mark.parametrize("value, counted", [
    (999.0, True), (998.9, False), (-1000.0, True), (1e6, True), (0.0, False),
], ids=["at-threshold", "below", "negative", "huge", "zero"])
def test_sentinel_threshold_boundary(value, counted):
    df = pd.DataFrame({"pt_bin": [0, 0], "label": [1, 0], "a": [value, 0.0]})
    table = sentinel_table(df, ["a"], by=["pt_bin"])
    assert table.loc[table["class"] == "signal", "n_sentinel"].iloc[0] == int(counted)


def test_sentinel_threshold_value():
    assert SENTINEL_THRESHOLD == 999.0
