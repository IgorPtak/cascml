import numpy as np
import pytest
import uproot

from cascml.data import DataError, list_trees, resolve_inputs


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
