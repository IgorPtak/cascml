"""Reading O2camlcandidates from AO2D files and preparing training samples."""

import glob
from pathlib import Path

import uproot


class DataError(Exception):

    """Raised when input data cannot be found or does not match the config."""


def resolve_inputs(patterns, base_dir):
    """Expand file paths/globs into a sorted list of existing files.

    Relative patterns are resolved against `base_dir`.
    """
    files = []
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        matches = sorted(glob.glob(str(path)))
        if not matches:
            raise DataError(f"no input files match '{pattern}' (resolved to {path})")
        files.extend(Path(m).resolve() for m in matches)
    return files


def list_trees(files, df_prefix, tree):
    """Return uproot paths 'file:DF_xxx/tree' for every DF directory in `files`."""
    paths = []
    for file in files:
        with uproot.open(file) as f:
            dirs = [k for k in f.keys(recursive=False, cycle=False) if k.startswith(df_prefix)]
            if not dirs:
                raise DataError(f"{file}: no directories starting with '{df_prefix}'")
            for d in dirs:
                if f"{d}/{tree}" not in f:
                    raise DataError(f"{file}: directory '{d}' has no tree '{tree}'")
                paths.append(f"{file}:{d}/{tree}")
    return paths
