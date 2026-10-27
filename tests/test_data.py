import pytest

from cascml.data import DataError, resolve_inputs


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
