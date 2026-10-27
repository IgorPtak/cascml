# cascml

Training Ξ/Ω classifiers for `cascademlselection` in O2Physics. Input: the `O2camlcandidates` table from `cascadeMLSelectionTreeCreator` (MC AO2D).

## Setup

```bash
uv sync
```

## Data

AO2D files are not in the repository. By default the config looks for them in `../data/` (next to the repository); pass another location with `--input`.

## Building samples

```bash
uv run cascml -c configs/cascade_analysis.yml [--input FILE.root ...] [--output DIR]
```

Output goes to `output/<case>/<species>/`: a `manifest.json` and one Parquet file per pT bin (12 features with O2 names and order, `pt`, `charge`, `label`, `is_test`).

```python
from cascml.data import read_manifest, read_sample

m = read_manifest("output/xi_test_hyperloop/xi")
for entry in m["bins"]:
    df, meta = read_sample(entry["file"], m["sample_hash"])
    train, test = df[~df.is_test], df[df.is_test]
```

## Tests

```bash
uv run pytest
```

Tests that need the Hyperloop file are skipped when it is missing; set its path with `CASCML_TEST_FILE`.
