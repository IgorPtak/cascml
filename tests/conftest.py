import os
from pathlib import Path

import numpy as np
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
