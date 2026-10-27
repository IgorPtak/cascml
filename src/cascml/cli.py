"""Command-line entry point and configuration validation for cascml."""

import argparse
import sys
from pathlib import Path

import yaml

from cascml.data import DataError, list_trees, prepare_species, resolve_inputs


class ConfigError(Exception):
    """Raised when the analysis or model configuration is invalid."""


# Order and names must match enum InputFeaturesCasc
# in O2Physics/PWGLF/Utils/CascadeMlResponse.h 
O2_FEATURES = (
    "cascradius",
    "v0radius",
    "casccosPA",
    "v0cosPA",
    "dcapostopv",
    "dcanegtopv",
    "dcabachtopv",
    "dcacascdaughters",
    "dcaV0daughters",
    "dcav0topv",
    "bachBaryonCosPA",
    "bachBaryonDCAxyToPV",
)

ALLOWED_TRANSFORMS = {"cos"}

TOP_LEVEL_KEYS = {
    "case",
    "input",
    "features",
    "species",
    "classes",
    "preselection",
    "tree_creator_cuts",
    "split",
    "guards",
    "validation",
    "export",
}


def load_yaml(path):
    """Load a YAML file into a dict, raising ConfigError on problems."""
    path = Path(path) 
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def validate_analysis(cfg):
    """Check the analysis config; raise ConfigError on the first problem."""
    if _contains_key(cfg, "mask_values"):
        raise ConfigError(
            "'mask_values' is forbidden: O2 passes raw sentinel values to the model, "
            "so replacing them with NaN in training breaks inference"
        )
    unknown = set(cfg) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)}")
    missing = TOP_LEVEL_KEYS - set(cfg)
    if missing:
        raise ConfigError(f"missing top-level keys: {sorted(missing)}")
    
    features = cfg["features"]
    if not isinstance(features, dict) or not features:
        raise ConfigError("'features' must be a non-empty mapping")
    for name, spec in features.items():
        if name not in O2_FEATURES:
            raise ConfigError(
                f"feature '{name}' is not in InputFeaturesCasc; allowed: {list(O2_FEATURES)}"
            )

        if not isinstance(spec, dict) or "column" not in spec:
            raise ConfigError(f"feature '{name}' must be a mapping with a 'column' key")
        
        transform = spec.get("transform")
        if transform is not None and transform not in ALLOWED_TRANSFORMS:
            raise ConfigError(
                f"feature '{name}': unknown transform '{transform}'; "
                f"allowed: {sorted(ALLOWED_TRANSFORMS)}"
            )
            
    species = cfg["species"]
    if not isinstance(species, dict) or not species: 
        raise ConfigError("'species' must be a non-empty mapping")
    for sp_name, sp in species.items():
        if not isinstance(sp, dict):
            raise ConfigError(f"species '{sp_name}' must be a mapping")
        bit = sp.get("mass_window_bit")
        if isinstance(bit, bool) or bit not in (0, 1):
            raise ConfigError(f"species '{sp_name}': mass_window_bit must be 0 (Xi) or 1 (Omega)")
    
        bins = sp.get("pt_bins")
        if not isinstance(bins, list) or len(bins) < 2:
            raise ConfigError(f"species '{sp_name}': pt_bins must be a list of at least 2 edges")
        if not all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in bins):
            raise ConfigError(f"species '{sp_name}': pt_bins must contain only numbers")
        if bins[0] < 0 or not all(a < b for a, b in zip(bins, bins[1:])):
            raise ConfigError(f"species '{sp_name}': pt_bins must be non-negative and strictly increasing, got {bins}")

        flags = sp.get("signal_flags")
        if not isinstance(flags, dict) or set(flags) != {"minus", "plus"}:
            raise ConfigError(f"species '{sp_name}': signal_flags must map exactly 'minus' and 'plus' to MC flag columns")
        if not isinstance(sp.get("split_by_charge"), bool):
            raise ConfigError(f"species '{sp_name}': split_by_charge must be true or false")
        
    classes = cfg["classes"]
    if not isinstance(classes, list) or [c.get("name") for c in classes if isinstance(c, dict)] != ["bkg", "signal"]:
        raise ConfigError("classes must be exactly [bkg, signal] in this order (index 1 = signal in ONNX output)")
    for c in classes:
        if c.get("source") != "mc":
            raise ConfigError(f"class '{c['name']}': only source 'mc' is implemented, got {c.get('source')!r}")

    _require_fraction(cfg["split"].get("test_size"), "split.test_size")
    _require_positive_int(cfg["guards"].get("min_signal_train"), "guards.min_signal_train")
    _require_positive_int(cfg["guards"].get("min_signal_test"), "guards.min_signal_test")

    bootstrap = cfg["validation"].get("bootstrap", {})
    _require_positive_int(bootstrap.get("n"), "validation.bootstrap.n")
    _require_fraction(bootstrap.get("ci"), "validation.bootstrap.ci")
    for wp in cfg["validation"].get("working_points", []):
        _require_fraction(wp, "validation.working_points")
            

def model_params_for(params, species):
    """Return XGBoost hyperparameters for `species`: defaults overridden by per_species."""
    if not isinstance(params.get("default"), dict):
        raise ConfigError("model params: 'default' section is required")
    overrides = params.get("per_species", {}).get(species) or {}
    return {**params["default"], **overrides}


def _contains_key(obj, key):
    """Return True if `key` appears in any mapping nested anywhere in `obj`."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_contains_key(value, key) for value in obj.values())
    if isinstance(obj, list):
        return any(_contains_key(item, key) for item in obj)
    return False


def _require_fraction(value, where):
    """Raise ConfigError unless `value` is a number strictly between 0 and 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ConfigError(f"{where} must be a number in (0, 1), got {value!r}")


def _require_positive_int(value, where):
    """Raise ConfigError unless `value` is an integer > 0."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where} must be a positive integer, got {value!r}")


def main(argv=None):
    """Entry point for the `cascml` command."""
    parser = argparse.ArgumentParser(prog="cascml", description="Train BDT models for cascademlselection in O2.")
    parser.add_argument("-c", "--config", required=True, help="analysis config (YAML)")
    parser.add_argument(
        "--model-params",
        help="XGBoost hyperparameters (YAML); default: model_params.yml next to the config",
    )
    parser.add_argument(
        "--input", nargs="+", metavar="FILE",
        help="override input.files from the config (paths relative to the current directory)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_yaml(args.config)
        validate_analysis(cfg)
        model_params = args.model_params or Path(args.config).parent / "model_params.yml"
        params = load_yaml(model_params)
        for sp_name in cfg["species"]:
            model_params_for(params, sp_name)
        if args.input:
            input_files = resolve_inputs(args.input, Path.cwd())
        else:
            input_files = resolve_inputs(cfg["input"]["files"], Path(args.config).parent)
        trees = list_trees(input_files, cfg["input"]["df_prefix"], cfg["input"]["tree"])
        samples = {sp_name: prepare_species(trees, cfg, sp_name) for sp_name in cfg["species"]}
    except (ConfigError, DataError) as err:
        print(f"cascml: {err}", file=sys.stderr)
        return 1

    print(f"case: {cfg['case']}")
    print(f"input files ({len(input_files)}): {', '.join(str(f) for f in input_files)}")
    print(f"trees: {len(trees)}")
    print(f"features ({len(cfg['features'])}): {', '.join(cfg['features'])}")
    for sp_name, sp in cfg["species"].items():
        print(f"species {sp_name}: pt_bins={sp['pt_bins']}, split_by_charge={sp['split_by_charge']}")
        df = samples[sp_name]
        edges = sp["pt_bins"]
        for b in range(len(edges) - 1):
            group = df[df["pt_bin"] == b]
            train, test = group[~group["is_test"]], group[group["is_test"]]
            print(
                f"  pT [{edges[b]}, {edges[b + 1]}): "
                f"train sig/bkg {int(train['label'].sum())}/{int((train['label'] == 0).sum())}, "
                f"test sig/bkg {int(test['label'].sum())}/{int((test['label'] == 0).sum())}"
            )

    return 0
