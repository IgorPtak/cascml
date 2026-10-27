"""Command-line entry point and configuration validation for cascml."""

import argparse
import sys
from pathlib import Path

import yaml

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
            
    if not isinstance(sp.get("split_by_charge"), bool):
        raise ConfigError(f"species '{sp_name}': split_by_charge must be true or false")


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
