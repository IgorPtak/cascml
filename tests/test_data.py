import numpy as np
import pandas as pd
import pytest
import uproot

from conftest import TREE_DEFAULTS
from cascml.data import (
    DataError,
    list_trees,
    load_species,
    required_columns,
    resolve_inputs,
    select_species,
    to_model_frame,
)


def test_relative_path_resolved_against_base_dir(tmp_path):
    (tmp_path / "a.root").touch()
    assert resolve_inputs(["a.root"], tmp_path) == [(tmp_path / "a.root").resolve()]


def test_glob_is_expanded_and_sorted(tmp_path):
    (tmp_path / "b.root").touch()
    (tmp_path / "a.root").touch()
    assert resolve_inputs(["*.root"], tmp_path) == [
        (tmp_path / "a.root").resolve(),
        (tmp_path / "b.root").resolve(),
    ]


def test_absolute_path_ignores_base_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.root").touch()
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    assert resolve_inputs([str(data_dir / "a.root")], other_dir) == [(data_dir / "a.root").resolve()]


def test_result_has_no_parent_references(tmp_path):
    sub = tmp_path / "configs"
    sub.mkdir()
    (tmp_path / "a.root").touch()
    [path] = resolve_inputs(["../a.root"], sub)
    assert ".." not in path.parts


def test_missing_input_raises(tmp_path):
    with pytest.raises(DataError, match="no input files"):
        resolve_inputs(["nie_ma_*.root"], tmp_path)


@pytest.fixture
def fake_ao2d(tmp_path):
    path = tmp_path / "AO2D.root"
    with uproot.recreate(path) as f:
        f["DF_1/O2camlcandidates"] = {"fPt": np.array([1.0, 2.0], dtype=np.float32)}
        f["DF_2/O2camlcandidates"] = {"fPt": np.array([3.0], dtype=np.float32)}
        f["parentFiles"] = "not a DF"
    return path


def test_list_trees_finds_dfs_and_skips_other_keys(fake_ao2d):
    assert list_trees([fake_ao2d], "DF_", "O2camlcandidates") == [
        f"{fake_ao2d}:DF_1/O2camlcandidates",
        f"{fake_ao2d}:DF_2/O2camlcandidates",
    ]


def test_list_trees_raises_without_df_dirs(fake_ao2d):
    with pytest.raises(DataError, match="no directories starting with 'XX_'"):
        list_trees([fake_ao2d], "XX_", "O2camlcandidates")


def test_list_trees_raises_without_tree(fake_ao2d):
    with pytest.raises(DataError, match="no tree 'O2nope'"):
        list_trees([fake_ao2d], "DF_", "O2nope")


def test_list_trees_output_is_readable_by_uproot(fake_ao2d):
    # Contract with the next step: the paths must be accepted by uproot.iterate.
    paths = list_trees([fake_ao2d], "DF_", "O2camlcandidates")
    assert sum(len(chunk["fPt"]) for chunk in uproot.iterate(paths, ["fPt"], library="np")) == 3


def test_hyperloop_file_has_192_dfs(hyperloop_file):
    assert len(list_trees([hyperloop_file], "DF_", "O2camlcandidates")) == 192


XI = {"mass_window_bit": 0, "signal_flags": {"minus": "fIsXiMinus", "plus": "fIsXiPlus"}}
OMEGA = {"mass_window_bit": 1, "signal_flags": {"minus": "fIsOmegaMinus", "plus": "fIsOmegaPlus"}}


@pytest.fixture
def mixed_chunk():
    # One row per case that matters for the species selection.
    return pd.DataFrame(
        {
            "fMassWindows": np.array([1, 3, 2, 1], dtype=np.uint8),
            "fIsXiMinus": [True, False, True, False],
            "fIsXiPlus": [False, True, False, False],
            "fIsOmegaMinus": [False, False, False, False],
            "fIsOmegaPlus": [False, False, False, False],
        },
        index=["xi_only", "both_windows", "omega_only", "xi_bkg"],
    )


def test_select_xi(mixed_chunk):
    out = select_species(mixed_chunk, XI)
    assert list(out.index) == ["xi_only", "both_windows", "xi_bkg"]
    assert list(out["label"]) == [1, 1, 0]


def test_select_omega_keeps_true_xi_as_background(mixed_chunk):
    out = select_species(mixed_chunk, OMEGA)
    assert list(out.index) == ["both_windows", "omega_only"]
    assert list(out["label"]) == [0, 0]


def test_required_columns(cfg):
    columns = required_columns(cfg, "xi")
    assert columns[:12] == [spec["column"] for spec in cfg["features"].values()]
    assert set(columns[12:]) == {"fPt", "fCharge", "fMassWindows", "fIsXiMinus", "fIsXiPlus"}


def test_model_frame_columns_and_types(cfg):
    chunk = pd.DataFrame({col: [default] for col, (_, default) in TREE_DEFAULTS.items()}).assign(label=1)
    out = to_model_frame(chunk, cfg["features"])
    assert list(out.columns) == list(cfg["features"]) + ["pt", "charge", "label"]
    assert all(out[name].dtype == np.float32 for name in cfg["features"])


def test_model_frame_applies_cos(cfg):
    angles = np.array([0.0, 0.1, 0.45], dtype=np.float32)
    chunk = pd.DataFrame({col: [default] * 3 for col, (_, default) in TREE_DEFAULTS.items()}).assign(label=0)
    chunk["fCascPA"] = angles
    out = to_model_frame(chunk, cfg["features"])
    assert out["casccosPA"][0] == 1.0
    assert np.isclose(out["casccosPA"], np.cos(angles.astype(np.float64)), atol=1e-6).all()
    assert (out["cascradius"] == np.float32(2.0)).all()


def test_load_species_reads_all_dfs(make_ao2d, cfg):
    path = make_ao2d(
        [
            [{"fMassWindows": 1, "fIsXiMinus": True}, {"fMassWindows": 2}],
            [{"fMassWindows": 3, "fIsXiPlus": True, "fCharge": 1}, {"fMassWindows": 1}],
        ]
    )
    out = load_species(list_trees([path], "DF_", "O2camlcandidates"), cfg, "xi")
    assert list(out["label"]) == [1, 1, 0]
    assert list(out["charge"]) == [-1, 1, -1]
    assert list(out.index) == [0, 1, 2]


def test_load_species_reports_missing_column(make_ao2d, cfg):
    path = make_ao2d([[{}]], drop=("fCascRadius",))
    with pytest.raises(DataError, match="fCascRadius"):
        load_species(list_trees([path], "DF_", "O2camlcandidates"), cfg, "xi")


def test_hyperloop_xi_sample(hyperloop_file, cfg):
    out = load_species(list_trees([hyperloop_file], "DF_", "O2camlcandidates"), cfg, "xi")
    assert len(out) == 12050
    assert out["label"].sum() == 5450
    assert set(out.loc[out["label"] == 1, "charge"]) == {-1, 1}
    # Tree-creator cut cos(PA) >= 0.9 must survive the angle -> cos transform.
    assert out["casccosPA"].min() >= np.float32(0.9)
