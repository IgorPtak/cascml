"""Reading O2camlcandidates from AO2D files and preparing training samples."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split


class DataError(Exception):
    """Raised when input data cannot be found or does not match the config."""


BASE_COLUMNS = ("fPt", "fCharge", "fMassWindows")


def resolve_inputs(patterns, base_dir):
    """Expand file paths/globs into a sorted list of existing files.

    Relative patterns are resolved against `base_dir`.
    """
    files = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        matches = sorted(glob.glob(str(path)))
        if not matches:
            raise DataError(f"no input files match '{pattern}' (resolved to {path})")
        files.extend(Path(m).resolve() for m in matches)
    return files


def list_trees(files, df_prefix, tree):
    """Return uproot paths 'file:DF_xxx/tree' for every DF directory in `files`."""
    paths = []
    for file in files:
        with uproot.open(file) as f:
            dirs = [k for k in f.keys(recursive=False, cycle=False) if k.startswith(df_prefix)]
            if not dirs:
                raise DataError(f"{file}: no directories starting with '{df_prefix}'")
            for d in dirs:
                if f"{d}/{tree}" not in f:
                    raise DataError(f"{file}: directory '{d}' has no tree '{tree}'")
                paths.append(f"{file}:{d}/{tree}")
    return paths


def required_columns(cfg, species):
    """Tree columns needed to build the sample for `species`."""
    sp = cfg["species"][species]
    columns = [spec["column"] for spec in cfg["features"].values()]
    columns += list(BASE_COLUMNS)
    columns += [sp["signal_flags"]["minus"], sp["signal_flags"]["plus"]]
    return columns


def check_columns(tree_path, columns):
    """Raise DataError if any of `columns` is missing in the tree."""
    with uproot.open(tree_path) as t:
        missing = [c for c in columns if c not in t.keys()]
    if missing:
        raise DataError(f"{tree_path}: missing columns {missing}")


def select_species(chunk, sp):
    """Keep candidates in the species mass window and add a 0/1 label."""
    in_window = (chunk["fMassWindows"] & (1 << sp["mass_window_bit"])) != 0
    chunk = chunk[in_window]
    label = chunk[sp["signal_flags"]["minus"]] | chunk[sp["signal_flags"]["plus"]]
    return chunk.assign(label=label.astype("int8"))


def to_model_frame(chunk, features):
    """Build the model input: O2 feature names in config order (float32) plus pt, charge, label."""
    out = {}
    for name, spec in features.items():
        values = chunk[spec["column"]].to_numpy(dtype=np.float32)
        if spec.get("transform") == "cos":
            values = np.cos(values)
        out[name] = values
    out["pt"] = chunk["fPt"].to_numpy(dtype=np.float32)
    out["charge"] = chunk["fCharge"].to_numpy(dtype=np.int8)
    out["label"] = chunk["label"].to_numpy()
    return pd.DataFrame(out)


def load_species(trees, cfg, species):
    """Read all trees chunk by chunk and return the model frame for `species`."""
    sp = cfg["species"][species]
    columns = required_columns(cfg, species)
    check_columns(trees[0], columns)
    frames = []
    for chunk in uproot.iterate(trees, columns, step_size=cfg["input"]["step_size"], library="pd"):
        frames.append(to_model_frame(select_species(chunk, sp), cfg["features"]))
    return pd.concat(frames, ignore_index=True)


def assign_pt_bins(df, edges):
    """Add `pt_bin` (index into `edges`) and drop candidates outside [edges[0], edges[-1])."""
    edges = np.asarray(edges, dtype=np.float32)
    idx = np.searchsorted(edges, df["pt"].to_numpy(), side="right") - 1
    inside = (idx >= 0) & (idx < len(edges) - 1)
    return df[inside].assign(pt_bin=idx[inside].astype(np.int8)).reset_index(drop=True)


def split_train_test(df, test_size, random_state):
    """Add `is_test`, stratified by label separately in every pT bin."""
    is_test = np.zeros(len(df), dtype=bool)
    for b, group in df.groupby("pt_bin"):
        counts = group["label"].value_counts()
        if len(counts) < 2 or counts.min() < 2:
            raise DataError(f"pT bin {b}: need >= 2 signal and >= 2 background candidates to split, got {counts.to_dict()}")
        _, test_idx = train_test_split(
            group.index, test_size=test_size, random_state=random_state, stratify=group["label"]
        )
        is_test[test_idx] = True
    return df.assign(is_test=is_test)


def check_guards(df, edges, guards, species):
    """Raise DataError if any pT bin is empty or has too little signal in train or test."""
    for b in range(len(edges) - 1):
        group = df[df["pt_bin"] == b]
        signal = group["label"] == 1
        n_train = int((signal & ~group["is_test"]).sum())
        n_test = int((signal & group["is_test"]).sum())
        if n_train < guards["min_signal_train"] or n_test < guards["min_signal_test"]:
            raise DataError(
                f"{species} pT [{edges[b]}, {edges[b + 1]}): signal train={n_train}, test={n_test}; "
                f"need >= {guards['min_signal_train']} / {guards['min_signal_test']}"
            )

def prepare_species(trees, cfg, species):
    """Full sample for `species`: selection, O2 features, pT bins, train/test split, statistics guards."""
    edges = cfg["species"][species]["pt_bins"]
    df = assign_pt_bins(load_species(trees, cfg, species), edges)
    df = split_train_test(df, cfg["split"]["test_size"], cfg["split"]["random_state"])
    check_guards(df, edges, cfg["guards"], species)
    return df
