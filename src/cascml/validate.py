"""Model-independent validation: scores joined with prepared samples by cand_id."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Key under which cascml stores score-file metadata in the Parquet schema.
SCORES_META_KEY = b"cascml_scores"


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
