"""Command-line entry point and configuration validation for cascml."""

import argparse
import sys
from pathlib import Path

import yaml

from cascml.data import SAMPLE_COLUMNS_EXTRA, DataError, list_trees, prepare_species, resolve_inputs, write_samples
from cascml.export import ExportError, export_species
from cascml.validate import AUC_CI_METHODS, ValidationError, validate_model
from cascml.train import train_species


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
    "meta_columns",
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

    meta_columns = cfg["meta_columns"]
    if not isinstance(meta_columns, dict):
        raise ConfigError("'meta_columns' must be a mapping (may be empty: {})")
    reserved = set(features) | set(SAMPLE_COLUMNS_EXTRA) | {"pt_bin"}
    for name, spec in meta_columns.items():
        if name in reserved:
            raise ConfigError(f"meta column '{name}' clashes with a feature or a sample column")
        if not isinstance(spec, dict) or set(spec) != {"column"}:
            raise ConfigError(f"meta column '{name}' must be a mapping with only a 'column' key")

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
        fraction = sp.get("bkg_fraction", 1.0)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            raise ConfigError(f"species '{sp_name}': bkg_fraction must be a number in (0, 1], got {fraction!r}")
        
    classes = cfg["classes"]
    if not isinstance(classes, list) or [c.get("name") for c in classes if isinstance(c, dict)] != ["bkg", "signal"]:
        raise ConfigError("classes must be exactly [bkg, signal] in this order (index 1 = signal in ONNX output)")
    for c in classes:
        if c.get("source") != "mc":
            raise ConfigError(f"class '{c['name']}': only source 'mc' is implemented, got {c.get('source')!r}")

    _require_fraction(cfg["split"].get("test_size"), "split.test_size")
    n_folds = cfg["split"].get("n_folds")
    _require_positive_int(n_folds, "split.n_folds")
    if n_folds < 2:
        raise ConfigError("split.n_folds must be at least 2 (one fold is used for validation)")
    _require_positive_int(cfg["guards"].get("min_signal_train"), "guards.min_signal_train")
    _require_positive_int(cfg["guards"].get("min_signal_test"), "guards.min_signal_test")

    onnx_dir = (cfg["export"] or {}).get("onnx_dir_name")
    if not isinstance(onnx_dir, str) or not onnx_dir or "/" in onnx_dir or onnx_dir in (".", ".."):
        raise ConfigError(f"export.onnx_dir_name must be a plain directory name, got {onnx_dir!r}")

    validation = cfg["validation"]
    bootstrap = validation.get("bootstrap") or {}
    if "ci" in bootstrap:
        raise ConfigError("validation.bootstrap.ci was moved to validation.auc_ci.level")
    auc_ci = validation.get("auc_ci")
    if not isinstance(auc_ci, dict) or auc_ci.get("method") not in AUC_CI_METHODS:
        raise ConfigError(f"validation.auc_ci.method must be one of {', '.join(AUC_CI_METHODS)}")
    _require_fraction(auc_ci.get("level"), "validation.auc_ci.level")
    if auc_ci["method"] == "bootstrap":
        _require_positive_int(bootstrap.get("n"), "validation.bootstrap.n")
    for wp in cfg["validation"].get("working_points", []):
        _require_fraction(wp, "validation.working_points")
            

def model_params_for(params, species):
    """Return XGBoost hyperparameters for `species`: defaults overridden by per_species."""
    if not isinstance(params.get("default"), dict):
        raise ConfigError("model params: 'default' section is required")
    overrides = params.get("per_species", {}).get(species) or {}
    merged = {**params["default"], **overrides}
    spw = merged.get("scale_pos_weight")
    if spw is not None and spw != "auto":
        if isinstance(spw, bool) or not isinstance(spw, (int, float)) or spw <= 0:
            raise ConfigError(f"model params for {species}: scale_pos_weight must be 'auto' or a positive number")
    return merged


def run_prepare(args):
    """`cascml prepare`: validate the config, read AO2D files and write Parquet samples."""
    cfg = load_yaml(args.config)
    validate_analysis(cfg)
    if args.input:
        input_files = resolve_inputs(args.input, Path.cwd())
    else:
        input_files = resolve_inputs(cfg["input"]["files"], Path(args.config).parent)
    trees = list_trees(input_files, cfg["input"]["df_prefix"], cfg["input"]["tree"])
    samples = {sp_name: prepare_species(trees, cfg, sp_name) for sp_name in cfg["species"]}
    output_root = _output_root(args)
    manifests = {
        sp_name: write_samples(df, cfg, sp_name, input_files, output_root) for sp_name, df in samples.items()
    }
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
        print(f"  samples: {manifests[sp_name]}")
    return 0


def run_validate(args):
    """`cascml validate`: metrics of one model for every species, from its score files."""
    cfg = load_yaml(args.config)
    validate_analysis(cfg)
    output_root = _output_root(args)
    for sp_name in cfg["species"]:
        path, doc = validate_model(output_root / cfg["case"] / sp_name, args.model, cfg)
        print(f"species {sp_name}, model {args.model}: {path}")
        edges = doc["pt_bins"]
        for row in doc["auc_test"]:
            b = row["pt_bin"]
            print(
                f"  pT [{edges[b]}, {edges[b + 1]}): test AUC {row['auc']:.3f} "
                f"[{row['auc_lo']:.3f}, {row['auc_hi']:.3f}], "
                f"{row['n_signal']} signal / {row['n_bkg']} bkg"
            )
        print(f"  plots: {path.parent / 'plots'}")
    return 0


def run_train(args):
    """`cascml train`: train XGBoost per species and pT bin and write scores.parquet."""
    cfg = load_yaml(args.config)
    validate_analysis(cfg)
    params_path = args.params or (Path(args.config).resolve().parent / "model_params.yml")
    params_cfg = load_yaml(params_path)
    output_root = _output_root(args)
    seed = cfg["split"]["random_state"]
    for sp_name in cfg["species"]:
        xgb_params = model_params_for(params_cfg, sp_name)
        species_dir = output_root / cfg["case"] / sp_name
        out_path, stats = train_species(species_dir, xgb_params, args.model, seed)
        print(f"species {sp_name}, model {args.model}: {out_path}")
        for s in stats:
            spw = s["scale_pos_weight"]
            spw_text = "default" if spw is None else f"{spw:.1f}"
            print(
                f"  pT [{s['pt_lo']}, {s['pt_hi']}): "
                f"best_iter={s['best_iteration']}, val_auc={s['best_score']:.4f}, scale_pos_weight={spw_text}"
            )
    return 0


def run_export(args):
    """`cascml export`: ONNX files per species and pT bin, verified against the validated scores."""
    cfg = load_yaml(args.config)
    validate_analysis(cfg)
    output_root = _output_root(args)
    for sp_name in cfg["species"]:
        species_dir = output_root / cfg["case"] / sp_name
        records = export_species(
            species_dir, args.model, cfg["export"]["onnx_dir_name"], args.opset, cfg["split"]["random_state"]
        )
        print(f"species {sp_name}, model {args.model}: {species_dir / 'models' / args.model / cfg['export']['onnx_dir_name']}")
        for r in records:
            print(
                f"  {r['file']}: parity max |diff| {r['max_abs_diff']:.2g} on {r['n_parity']} candidates "
                f"({r['n_parity_sentinel']} with sentinels), {r['us_per_candidate_batch1']:.1f} us/candidate"
            )
    return 0


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


def _output_root(args):
    """--output if given, else output/ next to the directory containing the config."""
    return Path(args.output) if args.output else Path(args.config).resolve().parent.parent / "output"


def main(argv=None):
    """Entry point for the `cascml` command."""
    parser = argparse.ArgumentParser(prog="cascml", description="Train BDT models for cascademlselection in O2.")
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", required=True, help="analysis config (YAML)")
    common.add_argument(
        "--output",
        help="output root directory; default: output/ next to the directory containing the config",
    )

    prepare = commands.add_parser("prepare", parents=[common], help="build Parquet samples from AO2D files")
    prepare.add_argument(
        "--input", nargs="+", metavar="FILE",
        help="override input.files from the config (paths relative to the current directory)",
    )
    prepare.set_defaults(func=run_prepare)

    validate = commands.add_parser("validate", parents=[common], help="compute validation metrics of one model")
    validate.add_argument(
        "--model", required=True,
        help="model name; scores are read from <output>/<case>/<species>/models/<model>/scores.parquet",
    )
    validate.set_defaults(func=run_validate)
    
    train = commands.add_parser("train", parents=[common], help="train XGBoost models on prepared samples")
    train.add_argument("--model", required=True, help="model name (saved under models/<model>/)")
    train.add_argument(
        "-p", "--params",
        help="model params YAML (default: model_params.yml next to the analysis config)",
    )
    train.set_defaults(func=run_train)

    export = commands.add_parser("export", parents=[common], help="export a trained model to ONNX for O2Physics")
    export.add_argument("--model", required=True, help="model name (read from models/<model>/)")
    export.add_argument("--opset", type=int, default=15, help="ONNX opset (default: 15)")
    export.set_defaults(func=run_export)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DataError, ValidationError, ExportError) as err:
        print(f"cascml: {err}", file=sys.stderr)
        return 1
