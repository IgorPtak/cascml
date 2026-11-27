import copy

import numpy as np

from cascml.plots import PLOTS, write_plots
from cascml.validate import compute_metrics


def test_write_plots_with_a_single_pt_bin(cfg, make_sample, tmp_path):
    # Many bins are covered by test_validate; one bin makes fig.subplots return a bare Axes, not an array.
    cfg = copy.deepcopy(cfg)
    cfg["species"]["xi"]["pt_bins"] = [0.5, 10.0]
    df = make_sample(cfg, signal_shift=0.5)
    df["score"] = np.random.default_rng(1).uniform(size=len(df))
    metrics = compute_metrics(df, list(cfg["features"]), n_boot=20, ci=0.95, working_points=[0.8])
    paths = write_plots(df, {"model": "m", "pt_bins": [0.5, 10.0], **metrics}, tmp_path)
    assert sorted(p.name for p in paths) == sorted(f"{name}.png" for name in PLOTS)
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in paths)
