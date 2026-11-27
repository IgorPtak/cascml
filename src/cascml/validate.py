"""Model-independent validation: scores joined with prepared samples by cand_id."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.metrics import roc_auc_score

from cascml.data import read_manifest, read_sample
from cascml.plots import write_plots

# Key under which cascml stores score-file metadata in the Parquet schema.
SCORES_META_KEY = b"cascml_scores"
SENTINEL_THRESHOLD = 999.0
AUC_CI_METHODS = ("delong", "bootstrap")


class ValidationError(Exception):
    """Raised when scores do not match the prepared samples."""
    

def scores_path(species_dir, model_name):
    """Agreed location of the score file of one model for one species."""
    return Path(species_dir) / "models" / model_name / "scores.parquet"


def write_scores(path, cand_id, score, sample_hash, model_name):
    """Write (cand_id, score) pairs with the sample hash and model name in the metadata."""
    df = pd.DataFrame({
        "cand_id": np.asarray(cand_id, dtype=np.int64),
        "score": np.asarray(score, dtype=np.float64),
    })
    table = pa.Table.from_pandas(df, preserve_index = False)
    meta = {"sample_hash": sample_hash, "model": model_name}
    table = table.replace_schema_metadata({**table.schema.metadata, SCORES_META_KEY: json.dumps(meta)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    

def read_scores(path):
    """Read a score file; return (DataFrame, metadata)."""
    table = pq.read_table(path)
    schema_meta = table.schema.metadata or {}
    if SCORES_META_KEY not in schema_meta:
        raise ValidationError(f"{path}: not a cascml score file (no '{SCORES_META_KEY.decode()}' metadata)")
    return table.to_pandas(), json.loads(schema_meta[SCORES_META_KEY])


def attach_scores(sample, scores, scores_meta, sample_hash):
    """Return `sample` with a `score` column, joined by cand_id. Raise ValidationError on any mismatch."""
    if scores_meta["sample_hash"] != sample_hash:
        raise ValidationError(
            f"scores were computed on samples with hash {scores_meta['sample_hash']}, "
            f"current samples have {sample_hash}; recompute the scores"
        )
    bad = ~np.isfinite(scores["score"].to_numpy(dtype=np.float64))
    if bad.any():
        raise ValidationError(f"{bad.sum()} scores are not finite (NaN or inf)")
    dup = scores["cand_id"].duplicated()
    if dup.any():
        raise ValidationError(f"{dup.sum()} cand_id values are duplicated in the scores, e.g. {scores['cand_id'][dup].iloc[0]}")
    unknown = ~scores["cand_id"].isin(sample["cand_id"])
    if unknown.any():
        raise ValidationError(f"{unknown.sum()} scored cand_id values are not in the sample, e.g. {scores['cand_id'][unknown].iloc[0]}")
    missing = ~sample["cand_id"].isin(scores["cand_id"])
    if missing.any():
        raise ValidationError(f"{missing.sum()} sample candidates are without a score, e.g. cand_id {sample['cand_id'][missing].iloc[0]}")
    return sample.merge(scores[["cand_id", "score"]], on="cand_id", how="left", validate="one_to_one")


def roc_auc(label, score):
    """Area under the ROC curve; label 1 = signal. Raise ValidationError if a class is missing."""
    label = np.asarray(label)
    n_signal = int((label == 1).sum())
    n_bkg = int((label == 0).sum())
    if n_signal == 0 or n_bkg == 0:
        raise ValidationError(f"AUC needs both classes, got {n_signal} signal and {n_bkg} background")
    return float(roc_auc_score(label, score))


def auc_table(df, by, ci_method=None, ci_level=0.95, n_boot=0, seed=0):
    """AUC and class counts for every group of `df` defined by the columns `by`.

    ci_method "delong" or "bootstrap" (n_boot replicas) adds the confidence interval (columns auc_lo, auc_hi).
    """
    rows = []
    for key, group in df.groupby(list(by), sort=True):
        row = {
            **dict(zip(by, key)),
            "n_signal": int((group["label"] == 1).sum()),
            "n_bkg": int((group["label"] == 0).sum()),
        }
        if ci_method == "delong":
            row["auc"], row["auc_lo"], row["auc_hi"] = delong_auc(group["label"], group["score"], ci_level)
        elif ci_method == "bootstrap":
            row["auc"], row["auc_lo"], row["auc_hi"] = bootstrap_auc(
                group["label"], group["score"], n_boot, ci_level, seed
            )
        elif ci_method is None:
            row["auc"] = roc_auc(group["label"], group["score"])
        else:
            raise ValidationError(f"unknown AUC interval method {ci_method!r}, expected one of {AUC_CI_METHODS}")
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_auc(label, score, n_boot, ci, seed=0):
    """AUC with a percentile bootstrap confidence interval; return (auc, lo, hi).

    Signal and background are resampled separately, so every replica keeps the class counts.
    """
    label = np.asarray(label)
    score = np.asarray(score, dtype=np.float64)
    auc = roc_auc(label, score)
    sig = np.flatnonzero(label == 1)
    bkg = np.flatnonzero(label == 0)
    rng = np.random.default_rng(seed)
    replicas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(sig, size=sig.size), rng.choice(bkg, size=bkg.size)])
        replicas[i] = roc_auc_score(label[idx], score[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(replicas, [alpha, 1.0 - alpha])
    return auc, float(lo), float(hi)


def delong_components(label, score):
    """AUC and DeLong structural components; return (auc, v10, v01).

    v10[i]: fraction of background below signal i, v01[j]: fraction of signal above background j,
    ties count 1/2. Midranks give all of them in O(N log N) (Sun & Xu 2014).
    """
    label = np.asarray(label)
    score = np.asarray(score, dtype=np.float64)
    roc_auc(label, score)  # raises ValidationError if a class is missing
    sig = score[label == 1]
    bkg = score[label == 0]
    m, n = sig.size, bkg.size
    rank_all = rankdata(np.concatenate([sig, bkg]))
    v10 = (rank_all[:m] - rankdata(sig)) / n
    v01 = 1.0 - (rank_all[m:] - rankdata(bkg)) / m
    return float(v10.mean()), v10, v01


def delong_auc(label, score, ci):
    """AUC with the DeLong normal confidence interval clipped to [0, 1]; return (auc, lo, hi)."""
    auc, v10, v01 = delong_components(label, score)
    # A class with a single candidate has no sample variance; its term is left out instead of giving NaN.
    var = sum(np.var(v, ddof=1) / v.size for v in (v10, v01) if v.size > 1)
    half = norm.ppf(0.5 + ci / 2.0) * np.sqrt(var)
    return auc, float(max(0.0, auc - half)), float(min(1.0, auc + half))


def thresholds_for_efficiency(df, by, targets):
    """Score threshold that keeps at least each target fraction of signal, for every group of `df`."""
    rows = []
    for key, group in df.groupby(list(by), sort=True):
        sig_scores = np.sort(group.loc[group["label"] == 1, "score"].to_numpy())[::-1]
        if sig_scores.size == 0:
            raise ValidationError(f"no signal in group {dict(zip(by, key))}, cannot set a threshold")
        for target in targets:
            # k-th highest signal score; the epsilon stops 0.55 * 100 = 55.00000000000001 from rounding up to 56
            k = int(np.ceil(target * sig_scores.size - 1e-9))
            rows.append({**dict(zip(by, key)), "target": target, "threshold": float(sig_scores[k - 1])})
    return pd.DataFrame(rows)


def efficiency_table(df, thresholds, by):
    """Signal and background efficiency of `score >= threshold` for every group of `df` and every target.

    `thresholds` comes from thresholds_for_efficiency; its group columns must be a subset of `by`,
    e.g. thresholds per pT bin applied separately to each charge.
    """
    keys = [c for c in thresholds.columns if c not in ("target", "threshold")]
    if not set(keys) <= set(by):
        raise ValidationError(f"threshold columns {keys} must be a subset of the grouping columns {list(by)}")
    merged = df.merge(thresholds, on=keys, how="inner")
    merged["passed"] = merged["score"] >= merged["threshold"]
    group_cols = list(by) + ["target"]
    rows = []
    for key, group in merged.groupby(group_cols, sort=True):
        sig = group["label"] == 1
        n_sig, n_bkg = int(sig.sum()), int((~sig).sum())
        if n_sig == 0 or n_bkg == 0:
            raise ValidationError(
                f"group {dict(zip(group_cols, key))} has {n_sig} signal and {n_bkg} background, need both"
            )
        eff_sig = float(group.loc[sig, "passed"].mean())
        eff_bkg = float(group.loc[~sig, "passed"].mean())
        rows.append({
            **dict(zip(group_cols, key)),
            "threshold": float(group["threshold"].iloc[0]),
            "n_signal": n_sig,
            "n_bkg": n_bkg,
            "eff_signal": eff_sig,
            "eff_signal_err": float(np.sqrt(eff_sig * (1 - eff_sig) / n_sig)),
            "eff_bkg": eff_bkg,
            "eff_bkg_err": float(np.sqrt(eff_bkg * (1 - eff_bkg) / n_bkg)),
        })
    return pd.DataFrame(rows)


def ks_train_test(df, by):
    """Two-sample KS test of the score: training folds (fold >= 1) vs test set, for signal and background separately."""
    rows = []
    for key, group in df.groupby(list(by), sort=True):
        for label, name in ((1, "signal"), (0, "bkg")):
            cls = group[group["label"] == label]
            train = cls.loc[cls["fold"] >= 1, "score"].to_numpy()
            test = cls.loc[cls["fold"] == -1, "score"].to_numpy()
            if train.size == 0 or test.size == 0:
                raise ValidationError(f"group {dict(zip(by, key))}, {name}: {train.size} train and {test.size} test candidates, need both")
            result = ks_2samp(train, test)
            rows.append({
                **dict(zip(by, key)),
                "class": name,
                "n_train": int(train.size),
                "n_test": int(test.size),
                "ks_stat": float(result.statistic),
                "p_value": float(result.pvalue),
            })
    return pd.DataFrame(rows)


def sentinel_table(df, features, by, threshold=SENTINEL_THRESHOLD):
    """Number and fraction of candidates with |feature| >= threshold, per group, for signal and background separately."""
    rows = []
    for key, group in df.groupby(list(by), sort=True):
        for label, name in ((1, "signal"), (0, "bkg")):
            cls = group[group["label"] == label]
            for feature in features:
                n_sentinel = int((cls[feature].abs() >= threshold).sum())
                rows.append({
                    **dict(zip(by, key)),
                    "class": name,
                    "feature": feature,
                    "n": len(cls),
                    "n_sentinel": n_sentinel,
                    "fraction": n_sentinel / len(cls) if len(cls) > 0 else 0.0,
                })
    return pd.DataFrame(rows)


def load_validation_frame(species_dir, model_name):
    """All pT bins of one species with a pt_bin column and the model score joined by cand_id.

    Return (frame, manifest, scores_meta).
    """
    species_dir = Path(species_dir)
    if not (species_dir / "manifest.json").is_file():
        raise ValidationError(f"no samples in {species_dir} (manifest.json missing); run cascml prepare first")
    path = scores_path(species_dir, model_name)
    if not path.is_file():
        raise ValidationError(f"no scores for model '{model_name}': {path} does not exist")
    manifest = read_manifest(species_dir)
    parts = []
    for b, entry in enumerate(manifest["bins"]):
        sample, _ = read_sample(entry["file"], manifest["sample_hash"])
        sample["pt_bin"] = b
        parts.append(sample)
    scores, scores_meta = read_scores(path)
    frame = attach_scores(pd.concat(parts, ignore_index=True), scores, scores_meta, manifest["sample_hash"])
    return frame, manifest, scores_meta


def compute_metrics(df, features, working_points, ci_method="delong", ci_level=0.95, n_boot=0, seed=0):
    """All validation metrics of one model; every table is a list of row dicts, ready for JSON."""
    test = df[df["fold"] == -1]
    train = df[df["fold"] >= 1]
    thresholds = thresholds_for_efficiency(df[df["fold"] == 0], by=["pt_bin"], targets=working_points)
    interval = {"ci_method": ci_method, "ci_level": ci_level, "n_boot": n_boot, "seed": seed}
    tables = {
        "auc_test": auc_table(test, ["pt_bin"], **interval),
        "auc_test_by_charge": auc_table(test, ["pt_bin", "charge"], **interval),
        "auc_train": auc_table(train, ["pt_bin"]),
        "working_points_test": efficiency_table(test, thresholds, ["pt_bin"]),
        "working_points_test_by_charge": efficiency_table(test, thresholds, ["pt_bin", "charge"]),
        "ks_train_test": ks_train_test(df, ["pt_bin"]),
        "sentinels": sentinel_table(df, features, ["pt_bin"]),
    }
    return {name: table.to_dict(orient="records") for name, table in tables.items()}


def validate_model(species_dir, model_name, cfg):
    """Compute all metrics of one model and write models/<model>/metrics.json; return (path, document)."""
    frame, manifest, scores_meta = load_validation_frame(species_dir, model_name)
    val = cfg["validation"]
    auc_ci = {"method": val["auc_ci"]["method"], "level": val["auc_ci"]["level"]}
    if auc_ci["method"] == "bootstrap":
        auc_ci.update(n_boot=val["bootstrap"]["n"], seed=cfg["split"]["random_state"])
    metrics = compute_metrics(
        frame, manifest["features"], val["working_points"],
        ci_method=auc_ci["method"], ci_level=auc_ci["level"],
        n_boot=auc_ci.get("n_boot", 0), seed=auc_ci.get("seed", 0),
    )
    doc = {
        "model": scores_meta["model"],
        "case": manifest["case"],
        "species": manifest["species"],
        "sample_hash": manifest["sample_hash"],
        "pt_bins": manifest["pt_bins"],
        "auc_ci": auc_ci,
        "working_points": val["working_points"],
        **metrics,
    }
    path = scores_path(species_dir, model_name).with_name("metrics.json")
    path.write_text(json.dumps(doc, indent=2) + "\n")
    write_plots(frame, doc, path.parent / "plots")
    return path, doc
