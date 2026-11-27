"""Export of trained models to ONNX for O2Physics cascademlselection, with a mandatory parity check."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxmltools
import onnxruntime as ort
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from cascml.data import bin_dir_name, read_manifest, read_sample
from cascml.train import feature_matrix
from cascml.validate import SENTINEL_THRESHOLD, read_scores, scores_path

PARITY_TOLERANCE = 1e-5
N_PARITY_TEST = 10_000


class ExportError(Exception):
    """Raised when a model cannot be exported or the exported model disagrees with the validated scores."""


def onnx_file_name(species, lo, hi):
    return f"{species}_{bin_dir_name(lo, hi)}.onnx"


def convert_bin(model_file, n_features, opset):
    """ONNX model of one pT bin: input 'input' float [None, n_features], last output probabilities [None, 2]."""
    clf = xgb.XGBClassifier()
    clf.load_model(model_file)
    return convert_xgboost(
        clf, initial_types=[("input", FloatTensorType([None, n_features]))], target_opset=opset
    )


def open_session(path):
    """onnxruntime session as on the Grid: CPU only, one thread."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def check_contract(session, n_features):
    """The tensor contract cascademlselection relies on; raise ExportError if it is broken."""
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape[1:] != [n_features]:
        found = [(i.name, i.type, i.shape) for i in inputs]
        raise ExportError(f"expected one float input [None, {n_features}], got {found}")
    # O2 reads the last output tensor; a ZipMap would make it a sequence of maps instead of a tensor.
    last = outputs[-1]
    if last.type != "tensor(float)" or last.shape[1:] != [2]:
        raise ExportError(f"expected last output float [None, 2] (no ZipMap), got {last.name} {last.type} {last.shape}")


def predict_signal(session, X):
    """Signal probability: column 1 of the last output, as read by O2."""
    return np.asarray(session.run(None, {session.get_inputs()[0].name: X})[-1])[:, 1]


def parity_rows(sample, features, seed):
    """Up to N_PARITY_TEST test candidates plus every candidate with a sentinel value (|x| >= 999)."""
    test_idx = np.flatnonzero(sample["fold"].to_numpy() == -1)
    rng = np.random.default_rng(seed)
    if test_idx.size > N_PARITY_TEST:
        test_idx = rng.choice(test_idx, size=N_PARITY_TEST, replace=False)
    sentinel_idx = np.flatnonzero((sample[features].abs() >= SENTINEL_THRESHOLD).any(axis=1).to_numpy())
    idx = np.union1d(test_idx, sentinel_idx)
    return sample.iloc[idx], int(sentinel_idx.size)


def latency_us(session, X, batch, repeats):
    """Mean time per candidate in microseconds, for batches of `batch` candidates, measured from Python.

    For comparing models with each other, not a prediction of the O2 task time: with batch 1 the Python
    call overhead dominates, while O2 calls onnxruntime from C++ (one candidate at a time).
    """
    X = X[:batch]
    predict_signal(session, X)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        predict_signal(session, X)
    return (time.perf_counter() - start) / (repeats * len(X)) * 1e6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_species(species_dir, model_name, onnx_dir_name, opset, seed):
    """Convert every pT bin of one model to ONNX and verify it against scores.parquet; return per-bin records."""
    species_dir = Path(species_dir)
    model_dir = species_dir / "models" / model_name
    meta_path = model_dir / "train_meta.json"
    if not (species_dir / "manifest.json").is_file():
        raise ExportError(f"no samples in {species_dir} (manifest.json is missing); run cascml prepare first")
    if not meta_path.is_file():
        raise ExportError(f"no trained model '{model_name}': {meta_path} is missing; run cascml train first")

    manifest = read_manifest(species_dir)
    train_meta = json.loads(meta_path.read_text())
    scores, scores_meta = read_scores(scores_path(species_dir, model_name))
    hashes = {"manifest": manifest["sample_hash"], "train_meta": train_meta["sample_hash"],
              "scores": scores_meta["sample_hash"]}
    if len(set(hashes.values())) != 1:
        raise ExportError(f"sample_hash mismatch {hashes}; retrain the model on the current samples")
    if train_meta.get("family") != "xgboost":
        raise ExportError(f"model family {train_meta.get('family')!r} cannot be exported, only 'xgboost'")
    if train_meta["features"] != manifest["features"]:
        raise ExportError("features in train_meta.json differ from the manifest")

    features = manifest["features"]
    score_by_id = scores.set_index("cand_id")["score"]
    out_dir = model_dir / onnx_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in manifest["bins"]:
        lo, hi = entry["pt_lo"], entry["pt_hi"]
        model_file = model_dir / bin_dir_name(lo, hi) / "model.ubj"
        if not model_file.is_file():
            raise ExportError(f"{model_file} is missing")
        onnx_path = out_dir / onnx_file_name(manifest["species"], lo, hi)
        onnx_path.write_bytes(convert_bin(model_file, len(features), opset).SerializeToString())

        session = open_session(onnx_path)
        check_contract(session, len(features))
        sample, _ = read_sample(entry["file"], manifest["sample_hash"])
        rows, n_sentinel = parity_rows(sample, features, seed)
        X = feature_matrix(rows, features)
        diff = np.abs(predict_signal(session, X) - score_by_id.loc[rows["cand_id"]].to_numpy())
        max_diff = float(diff.max())
        if not max_diff <= PARITY_TOLERANCE:
            worst = rows["cand_id"].iloc[int(diff.argmax())]
            raise ExportError(
                f"{onnx_path.name}: ONNX differs from scores.parquet by {max_diff:.3g} "
                f"(> {PARITY_TOLERANCE:g}), e.g. cand_id {worst}; export aborted"
            )
        records.append({
            "pt_lo": lo,
            "pt_hi": hi,
            "file": onnx_path.name,
            "sha256": sha256(onnx_path),
            "n_parity": len(rows),
            "n_parity_sentinel": n_sentinel,
            "max_abs_diff": max_diff,
            "us_per_candidate_batch1": latency_us(session, X, batch=1, repeats=200),
            "us_per_candidate_batch1024": latency_us(session, X, batch=1024, repeats=5),
        })
    meta = {
        "model": model_name,
        "species": manifest["species"],
        "sample_hash": manifest["sample_hash"],
        "features": features,  # input column order the O2 task must feed
        "input_name": "input",
        "signal_output": "last output tensor, column 1",
        "opset": opset,
        "versions": {"xgboost": xgb.__version__, "onnxmltools": onnxmltools.__version__,
                     "onnxruntime": ort.__version__},
        "bins": records,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))
    return records
