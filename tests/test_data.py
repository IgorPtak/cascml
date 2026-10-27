import numpy as np
import pandas as pd
import pytest
import uproot

from conftest import TREE_DEFAULTS
from cascml.data import (
    DataError,
    assign_pt_bins,
    check_guards,
    list_trees,
    load_species,
    prepare_species,
    read_manifest,
    read_sample,
    required_columns,
    resolve_inputs,
    sample_hash,
    select_species,
    split_train_test,
    to_model_frame,
    write_samples,
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


EDGES = [0.5, 1.5, 3.0, 10.0]


def o2_find_pt_bin(edges, pt):
    # Direct transcription of findPtBin in cascademlselection.cxx (std::upper_bound - 1, -1 outside).
    if pt < edges[0] or pt >= edges[-1]:
        return -1
    return next(i for i, e in enumerate(edges) if e > pt) - 1


def test_pt_bin_edges_are_lower_inclusive():
    df = pd.DataFrame({"pt": np.array([0.49, 0.5, 1.4999, 1.5, 9.99, 10.0], dtype=np.float32)})
    out = assign_pt_bins(df, EDGES)
    assert list(out["pt"]) == list(np.array([0.5, 1.4999, 1.5, 9.99], dtype=np.float32))
    assert list(out["pt_bin"]) == [0, 0, 1, 2]
    assert list(out.index) == [0, 1, 2, 3]


def test_pt_bins_match_o2_find_pt_bin():
    pt = np.random.default_rng(1).uniform(0.0, 12.0, 2000).astype(np.float32)
    out = assign_pt_bins(pd.DataFrame({"pt": pt}), EDGES)
    expected = [b for b in (o2_find_pt_bin(EDGES, p) for p in pt) if b >= 0]
    assert list(out["pt_bin"]) == expected


@pytest.fixture
def binned():
    # Two bins, 30% signal in each.
    rng = np.random.default_rng(2)
    n = 1000
    return pd.DataFrame(
        {
            "pt_bin": np.repeat([0, 1], n // 2).astype(np.int8),
            "label": (rng.random(n) < 0.3).astype(np.int8),
        }
    )


def test_split_is_stratified_per_bin(binned):
    out = split_train_test(binned, 0.3, 42)
    for _, group in out.groupby("pt_bin"):
        assert abs(group["is_test"].mean() - 0.3) < 0.01
        sig_frac_train = group.loc[~group["is_test"], "label"].mean()
        sig_frac_test = group.loc[group["is_test"], "label"].mean()
        assert abs(sig_frac_train - sig_frac_test) < 0.01


def test_split_is_reproducible(binned):
    a = split_train_test(binned, 0.3, 42)["is_test"]
    b = split_train_test(binned, 0.3, 42)["is_test"]
    assert (a == b).all()


def test_split_rejects_bin_without_both_classes():
    df = pd.DataFrame({"pt_bin": np.zeros(10, dtype=np.int8), "label": np.zeros(10, dtype=np.int8)})
    with pytest.raises(DataError, match="pT bin 0"):
        split_train_test(df, 0.3, 42)


GUARDS = {"min_signal_train": 5, "min_signal_test": 2}


def test_guards_pass_with_enough_signal():
    df = pd.DataFrame(
        {"pt_bin": [0] * 10, "label": [1] * 10, "is_test": [False] * 7 + [True] * 3}
    )
    check_guards(df, [0.5, 1.5], GUARDS, "xi")


def test_guards_report_bin_in_pt_units():
    df = pd.DataFrame(
        {"pt_bin": [0] * 4, "label": [1] * 4, "is_test": [False] * 3 + [True]}
    )
    with pytest.raises(DataError, match=r"xi pT \[0\.5, 1\.5\): signal train=3, test=1"):
        check_guards(df, [0.5, 1.5], GUARDS, "xi")


def test_guards_catch_empty_bin():
    df = pd.DataFrame({"pt_bin": [0] * 10, "label": [1] * 10, "is_test": [False] * 7 + [True] * 3})
    with pytest.raises(DataError, match=r"\[1\.5, 3\.0\): signal train=0"):
        check_guards(df, [0.5, 1.5, 3.0], GUARDS, "xi")


def test_hyperloop_xi_prepared(hyperloop_file, cfg):
    out = prepare_species(list_trees([hyperloop_file], "DF_", "O2camlcandidates"), cfg, "xi")
    assert len(out) == 12017
    assert out.groupby("pt_bin")["label"].sum().tolist() == [1825, 2936, 682]
    assert abs(out["is_test"].mean() - 0.3) < 0.01


@pytest.fixture
def prepared(cfg):
    # Minimal prepared sample: 2 rows per pT bin of the real config, one signal and one background.
    n_bins = len(cfg["species"]["xi"]["pt_bins"]) - 1
    rows = 2 * n_bins
    df = pd.DataFrame({name: np.full(rows, 0.5, dtype=np.float32) for name in cfg["features"]})
    return df.assign(
        pt=np.full(rows, 1.0, dtype=np.float32),
        charge=np.tile(np.array([-1, 1], dtype=np.int8), n_bins),
        label=np.tile(np.array([1, 0], dtype=np.int8), n_bins),
        is_test=np.tile([False, True], n_bins),
        pt_bin=np.repeat(np.arange(n_bins, dtype=np.int8), 2),
    )


INPUTS = ["/data/AO2D_1.root"]


def test_samples_roundtrip(prepared, cfg, tmp_path):
    manifest = read_manifest(write_samples(prepared, cfg, "xi", INPUTS, tmp_path).parent)
    assert [b["file"].parent.parent.name for b in manifest["bins"]] == ["pt_0.5_1.5", "pt_1.5_3", "pt_3_10"]
    df, meta = read_sample(manifest["bins"][0]["file"], manifest["sample_hash"])
    # Contract with other models: O2 feature order, then pt, charge, label, is_test; no pt_bin.
    assert list(df.columns) == list(cfg["features"]) + ["pt", "charge", "label", "is_test"]
    assert df["cascradius"].dtype == np.float32 and df["label"].dtype == np.int8
    assert (meta["pt_lo"], meta["pt_hi"], meta["input_files"]) == (0.5, 1.5, INPUTS)


def test_manifest_counts(prepared, cfg, tmp_path):
    manifest = read_manifest(write_samples(prepared, cfg, "xi", INPUTS, tmp_path).parent)
    counts = {k: v for k, v in manifest["bins"][0].items() if k.startswith("n_")}
    assert counts == {"n_train_signal": 1, "n_train_bkg": 0, "n_test_signal": 0, "n_test_bkg": 1}


def test_hash_depends_on_config_and_inputs(cfg):
    base = sample_hash(cfg, INPUTS)
    assert sample_hash(cfg, INPUTS) == base
    assert sample_hash(cfg, ["/data/AO2D_2.root"]) != base
    cfg["split"]["random_state"] = 7
    assert sample_hash(cfg, INPUTS) != base


def test_stale_sample_is_rejected(prepared, cfg, tmp_path):
    manifest = read_manifest(write_samples(prepared, cfg, "xi", INPUTS, tmp_path).parent)
    with pytest.raises(DataError, match="regenerate"):
        read_sample(manifest["bins"][0]["file"], expected_hash="0" * 16)
