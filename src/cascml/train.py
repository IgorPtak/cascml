"""XGBoost training per pT bin on prepared Parquet samples."""

from pathlib import Path

import numpy as np
import xgboost as xgb

from cascml.data import DataError, bin_dir_name, read_manifest, read_sample
from cascml.validate import scores_path, write_scores


def feature_matrix(frame, features):
    """Model input in manifest["features"] order, float32 (the O2 C++ side uses float)."""
    return frame[features].to_numpy(dtype=np.float32)


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
        X_val = feature_matrix(val, features)
        
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train,
            train["label"].to_numpy(),
            eval_set=[(X_val, val["label"].to_numpy())],
            verbose=False,
        )
        
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
        })
    
    out_path = scores_path(species_dir, model_name)
    write_scores(
        out_path,
        np.concatenate(all_cand_ids),
        np.concatenate(all_scores),
        manifest["sample_hash"],
        model_name,
    )
    return out_path, bin_stats