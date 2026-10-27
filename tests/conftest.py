import os
from pathlib import Path

import pytest

DEFAULT_TEST_FILE = Path(__file__).parent.parent.parent / "data" / "AO2D_derived_test_hyperloop.root"


@pytest.fixture
def hyperloop_file():
    path = Path(os.environ.get("CASCML_TEST_FILE", DEFAULT_TEST_FILE))
    if not path.is_file():
        pytest.skip(f"test file not found: {path} (set CASCML_TEST_FILE)")
    return path
