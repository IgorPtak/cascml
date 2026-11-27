import copy
from pathlib import Path

import pytest

from cascml.cli import ConfigError, main, model_params_for, validate_analysis

CONFIGS = Path(__file__).parent.parent / "configs"
CONFIG = CONFIGS / "cascade_analysis.yml"


def xi(c):
    return c["species"]["xi"]


def test_valid_config_passes(cfg):
    validate_analysis(cfg)


def test_delong_does_not_need_bootstrap_section(cfg):
    cfg["validation"].pop("bootstrap")
    validate_analysis(cfg)


@pytest.mark.parametrize(
    "mutate, message",
    [
        pytest.param(lambda c: c.update(typo_key=1), "unknown top-level keys", id="unknown-key"),
        pytest.param(lambda c: c.pop("guards"), "missing top-level keys", id="missing-section"),
        pytest.param(lambda c: xi(c).update(mask_values={}), "mask_values", id="mask-values-nested"),
        pytest.param(lambda c: c.update(mask_values={}), "mask_values", id="mask-values-top"),
        pytest.param(lambda c: c.update(features={}), "non-empty", id="features-empty"),
        pytest.param(
            lambda c: c["features"].update(cascRadius={"column": "fCascRadius"}),
            "not in InputFeaturesCasc",
            id="feature-unknown",
        ),
        pytest.param(lambda c: c["features"]["cascradius"].pop("column"), "'column' key", id="feature-no-column"),
        pytest.param(
            lambda c: c["features"]["casccosPA"].update(transform="acos"),
            "unknown transform",
            id="transform-unknown",
        ),
        pytest.param(lambda c: c.pop("meta_columns"), "missing top-level keys", id="meta-missing"),
        pytest.param(lambda c: c["meta_columns"].update(pt={"column": "fPt"}), "clashes", id="meta-clash-sample"),
        pytest.param(lambda c: c["meta_columns"].update(v0radius={"column": "fV0radius"}), "clashes", id="meta-clash-feature"),
        pytest.param(
            lambda c: c["meta_columns"].update(m={"column": "fMXi", "transform": "cos"}), "only a 'column'", id="meta-transform"
        ),
        pytest.param(lambda c: xi(c).update(mass_window_bit=2),"mass_window_bit", id="bit-out-of-range"),
        pytest.param(lambda c: xi(c).update(mass_window_bit=True), "mass_window_bit", id="bit-bool"),
        pytest.param(lambda c: xi(c).pop("mass_window_bit"), "mass_window_bit", id="bit-missing"),
        pytest.param(lambda c: xi(c).update(pt_bins=[0.5]), "at least 2 edges", id="pt-bins-single"),
        pytest.param(lambda c: xi(c).update(pt_bins=[0.5, "x"]), "only numbers", id="pt-bins-string"),
        pytest.param(lambda c: xi(c).update(pt_bins=[0.5, 3.0, 1.5]), "strictly increasing", id="pt-bins-unsorted"),
        pytest.param(lambda c: xi(c).update(pt_bins=[1.0, 1.0, 2.0]), "strictly increasing", id="pt-bins-duplicate"),
        pytest.param(lambda c: xi(c).update(pt_bins=[-1.0, 2.0]), "non-negative", id="pt-bins-negative"),
        pytest.param(lambda c: xi(c)["signal_flags"].pop("plus"), "signal_flags", id="flags-missing-plus"),
        pytest.param(lambda c: xi(c).update(split_by_charge="no"), "split_by_charge", id="split-by-charge-str"),
        pytest.param(lambda c: xi(c).update(split_by_charge=True), "not implemented", id="split-by-charge-true"),
        pytest.param(lambda c: c["classes"].reverse(), r"\[bkg, signal\]", id="classes-order"),
        pytest.param(lambda c: c["classes"].pop(), r"\[bkg, signal\]", id="classes-single"),
        pytest.param(
            lambda c: c["classes"][0].update(source="data_sidebands"),
            "only source 'mc'",
            id="classes-source",
        ),
        pytest.param(lambda c: c["split"].update(test_size=1.0), r"split\.test_size", id="test-size-one"),
        pytest.param(lambda c: c["split"].update(test_size="0.3"), r"split\.test_size", id="test-size-str"),
        pytest.param(lambda c: c["split"].pop("n_folds"), r"split\.n_folds", id="n-folds-missing"),
        pytest.param(lambda c: c["split"].update(n_folds=1), "at least 2", id="n-folds-one"),
        pytest.param(lambda c: c["guards"].update(min_signal_test=0), "min_signal_test", id="guard-zero"),
        pytest.param(lambda c: c["guards"].update(min_signal_train=True), "min_signal_train", id="guard-bool"),
        pytest.param(lambda c: c["export"].update(onnx_dir_name="../x"), "onnx_dir_name", id="onnx-dir-path"),
        pytest.param(lambda c: c["validation"].pop("auc_ci"), r"auc_ci\.method", id="auc-ci-missing"),
        pytest.param(lambda c: c["validation"]["auc_ci"].update(method="wilson"), r"auc_ci\.method", id="auc-ci-method"),
        pytest.param(lambda c: c["validation"]["auc_ci"].update(level=95), r"auc_ci\.level", id="auc-ci-level"),
        pytest.param(
            lambda c: (c["validation"]["auc_ci"].update(method="bootstrap"), c["validation"].pop("bootstrap")),
            r"bootstrap\.n",
            id="bootstrap-without-n",
        ),
        pytest.param(
            lambda c: c["validation"]["bootstrap"].update(ci=0.95), r"moved to validation\.auc_ci\.level", id="old-bootstrap-ci"
        ),
        pytest.param(lambda c: c["validation"].update(working_points=[0.8, 0]), "working_points", id="working-point"),
    ],
)
def test_invalid_config_raises(cfg, mutate, message):
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        validate_analysis(cfg)


def test_each_species_is_validated(cfg):
    # A broken species must be caught even when it is not the last one in the mapping.
    bad = copy.deepcopy(xi(cfg))
    bad["split_by_charge"] = "no"
    cfg["species"] = {"bad": bad, "xi": xi(cfg)}
    with pytest.raises(ConfigError, match="species 'bad'"):
        validate_analysis(cfg)


def test_model_params_override():
    params = {"default": {"max_depth": 4, "n_estimators": 300}, "per_species": {"omega": {"max_depth": 3}}}
    assert model_params_for(params, "omega") == {"max_depth": 3, "n_estimators": 300}
    assert model_params_for(params, "xi") == {"max_depth": 4, "n_estimators": 300}
    assert params["default"]["max_depth"] == 4


def test_model_params_require_default():
    with pytest.raises(ConfigError, match="default"):
        model_params_for({"per_species": {}}, "xi")


def test_main_succeeds_on_valid_config(capsys, tmp_path, hyperloop_file):
    args = ["prepare", "-c", str(CONFIG), "--input", str(hyperloop_file)]
    assert main(args + ["--output", str(tmp_path)]) == 0
    assert "species xi" in capsys.readouterr().out
    assert (tmp_path / "xi_test_hyperloop" / "xi" / "manifest.json").is_file()


def test_main_returns_1_on_missing_file(capsys):
    assert main(["prepare", "-c", "nie_ma.yml"]) == 1
    assert "config file not found" in capsys.readouterr().err


def test_main_requires_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "required: command" in capsys.readouterr().err


@pytest.mark.parametrize("value", [0, 1.5, -0.1, True, "0.2"])
def test_bkg_fraction_out_of_range(cfg, value):
    cfg["species"]["xi"]["bkg_fraction"] = value
    with pytest.raises(ConfigError, match="bkg_fraction"):
        validate_analysis(cfg)
