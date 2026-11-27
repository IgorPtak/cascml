"""XGBoost training per pT bin on prepared Parquet samples."""

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from cascml.data import DataError, bin_dir_name, read_manifest, read_sample
from cascml.validate import scores_path, write_scores


def feature_matrix(frame, features):
    """Model input in manifest["features"] order, float32 (the O2 C++ side uses float)."""
    return frame[features].to_numpy(dtype=np.float32)


def resolve_params(params, y_train):
    """XGBoost params for one pT bin; scale_pos_weight 'auto' becomes n_bkg / n_sig of y_train."""
    resolved = dict(params)
    if resolved.get("scale_pos_weight") == "auto":
        n_sig = int((y_train == 1).sum())
        if n_sig == 0:
            raise DataError("scale_pos_weight: auto needs signal in the training folds")
        resolved["scale_pos_weight"] = (len(y_train) - n_sig) / n_sig
    return resolved


def train_species(species_dir, xgb_params, model_name, seed):
    """Train one XGBClassifier per pT bin and write models/<model>/scores.parquet."""
    species_dir = Path(species_dir)
    if not (species_dir / "manifest.json").is_file():
        raise DataError(f"no samples in {species_dir} (manifest.json is missing); run cascml prepare first")

    manifest = read_manifest(species_dir)
    features = manifest["features"]
    params = {"random_state": seed, **xgb_params}
    
    all_cand_ids = []
    all_scores = []
    bin_stats = []
    
    for entry in manifest["bins"]:
        df, _ = read_sample(entry["file"], manifest["sample_hash"])
        train = df[df["fold"] > 0]
        val = df[df["fold"] == 0]
        
        X_train = feature_matrix(train, features)
        y_train = train["label"].to_numpy()
        X_val = feature_matrix(val, features)
        bin_params = resolve_params(params, y_train)

        clf = xgb.XGBClassifier(**bin_params)
        start = time.perf_counter()
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, val["label"].to_numpy())],
            verbose=False,
        )
        train_time_s = time.perf_counter() - start

        bin_name = bin_dir_name(entry["pt_lo"], entry["pt_hi"])
        model_file = species_dir / "models" / model_name / bin_name / "model.ubj"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        clf.save_model(model_file)
        
        scores = clf.predict_proba(feature_matrix(df, features))[:, 1]
        all_cand_ids.append(df["cand_id"].to_numpy())
        all_scores.append(scores)
        bin_stats.append({
            "pt_lo": entry["pt_lo"],
            "pt_hi": entry["pt_hi"],
            "best_iteration": getattr(clf, "best_iteration", None),
            "best_score": getattr(clf, "best_score", None),
            "scale_pos_weight": bin_params.get("scale_pos_weight"),
            "train_time_s": round(train_time_s, 3),
            "params_resolved": bin_params,
        })

    out_path = scores_path(species_dir, model_name)
    write_scores(
        out_path,
        np.concatenate(all_cand_ids),
        np.concatenate(all_scores),
        manifest["sample_hash"],
        model_name,
    )
    # Written last: its presence means training finished for every bin.
    meta = {
        "model": model_name,
        "family": "xgboost",
        "xgboost_version": xgb.__version__,
        "case": manifest["case"],
        "species": manifest["species"],
        "sample_hash": manifest["sample_hash"],
        "features": features,
        "pt_bins": manifest["pt_bins"],
        "seed": seed,
        "params": params,
        "bins": bin_stats,
    }
    out_path.with_name("train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_path, bin_stats