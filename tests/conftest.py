import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

from cascml.cli import load_yaml

REPO = Path(__file__).parent.parent
DEFAULT_TEST_FILE = REPO.parent / "data" / "AO2D_derived_test_hyperloop.root"

# Schema of O2camlcandidates restricted to the columns cascml reads, with neutral default values.
# Tests override only the columns they check.
TREE_DEFAULTS = {
    "fCascRadius": (np.float32, 2.0),
    "fV0radius": (np.float32, 5.0),
    "fCascPA": (np.float32, 0.05),
    "fV0PA": (np.float32, 0.03),
    "fDCApostopv": (np.float32, 0.2),
    "fDCAnegtopv": (np.float32, -0.2),
    "fDCAbachtopv": (np.float32, 0.1),
    "fDCACascDaughters": (np.float32, 0.1),
    "fDCAV0daughters": (np.float32, 0.05),
    "fDCAv0topv": (np.float32, 0.3),
    "fBachBaryonCosPA": (np.float32, 0.9),
    "fBachBaryonDCAxyToPV": (np.float32, 0.01),
    "fMXi": (np.float32, 1.3217),
    "fMOmega": (np.float32, 1.6725),
    "fCentrality": (np.float32, 50.0),
    "fPt": (np.float32, 1.0),
    "fCharge": (np.int32, -1),
    "fMassWindows": (np.uint8, 1),
    "fIsXiMinus": (np.bool_, False),
    "fIsXiPlus": (np.bool_, False),
    "fIsOmegaMinus": (np.bool_, False),
    "fIsOmegaPlus": (np.bool_, False),
}


@pytest.fixture
def cfg():
    return load_yaml(REPO / "configs" / "cascade_analysis.yml")


@pytest.fixture
def hyperloop_file():
    path = Path(os.environ.get("CASCML_TEST_FILE", DEFAULT_TEST_FILE))
    if not path.is_file():
        pytest.skip(f"test file not found: {path} (set CASCML_TEST_FILE)")
    return path


@pytest.fixture
def make_sample():
    """Factory: synthetic sample in the cascml sample format, for the xi species of the test config.

    signal_shift moves every feature of signal candidates, so that a model has something to learn.
    """

    def make(cfg, n_per_bin=400, seed=0, signal_shift=0.0):
        rng = np.random.default_rng(seed)
        edges = cfg["species"]["xi"]["pt_bins"]
        parts = []
        for b in range(len(edges) - 1):
            n = n_per_bin
            label = np.tile(np.array([0, 1], dtype=np.int8), n // 2)
            part = pd.DataFrame({
                f: (rng.normal(size=n) + signal_shift * label).astype(np.float32) for f in cfg["features"]
            })
            for name in cfg["meta_columns"]:
                part[name] = rng.uniform(size=n).astype(np.float32)
            part["pt"] = rng.uniform(edges[b], edges[b + 1], size=n).astype(np.float32)
            part["charge"] = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
            part["label"] = label
            part["fold"] = np.repeat(np.array([-1, 0, 1, 2, 3, 4, -1, 1], dtype=np.int8), n // 8)
            part["is_test"] = part["fold"] == -1
            part["pt_bin"] = b
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        df["cand_id"] = np.arange(len(df), dtype=np.int64) * 3 + 7
        return df

    return make


@pytest.fixture
def make_ao2d(tmp_path):
    """Factory: make_ao2d([[row, ...], [row, ...]]) writes one DF directory per list of rows."""

    def make(dfs, drop=()):
        path = tmp_path / "AO2D.root"
        with uproot.recreate(path) as f:
            for i, rows in enumerate(dfs, start=1):
                f[f"DF_{i}/O2camlcandidates"] = {
                    col: np.array([row.get(col, default) for row in rows], dtype=dtype)
                    for col, (dtype, default) in TREE_DEFAULTS.items()
                    if col not in drop
                }
        return path

    return make
