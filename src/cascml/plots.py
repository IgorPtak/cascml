"""Validation plots of one model, drawn from the joined frame and the metrics document."""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve


def _bin_label(edges, b):
    return f"{edges[b]:g} ≤ $p_\\mathrm{{T}}$ < {edges[b + 1]:g} GeV/$c$"


def plot_roc(df, doc):
    """ROC on the test set, one curve per pT bin: background efficiency (log) vs signal efficiency."""
    fig = Figure(figsize=(6, 5), layout="constrained")
    ax = fig.subplots()
    test = df[df["fold"] == -1]
    auc = {row["pt_bin"]: row for row in doc["auc_test"]}
    for b, group in test.groupby("pt_bin"):
        fpr, tpr, _ = roc_curve(group["label"], group["score"])
        a = auc[b]
        ax.plot(tpr, fpr, label=f"{_bin_label(doc['pt_bins'], b)}\nAUC {a['auc']:.3f} [{a['auc_lo']:.3f}, {a['auc_hi']:.3f}]")
    x = np.linspace(0.001, 1, 200)
    ax.plot(x, x, ls=":", color="grey", label="random classifier")
    ax.set(xlabel="signal efficiency", ylabel="background efficiency", yscale="log", xlim=(0, 1),
           title=f"{doc['model']}: ROC on test set")
    ax.legend(fontsize="small")
    return fig


def plot_score_train_test(df, doc, n_bins=50):
    """Score distributions per pT bin: training folds (filled) vs test set (points), signal and background."""
    edges = doc["pt_bins"]
    ks = {(row["pt_bin"], row["class"]): row for row in doc["ks_train_test"]}
    n = len(edges) - 1
    fig = Figure(figsize=(5 * n, 4.5), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, n))
    for b, ax in enumerate(axes):
        group = df[df["pt_bin"] == b]
        bins = np.linspace(group["score"].min(), group["score"].max(), n_bins + 1)
        centres = 0.5 * (bins[1:] + bins[:-1])
        for label, name, color in ((1, "signal", "tab:blue"), (0, "bkg", "tab:red")):
            cls = group[group["label"] == label]
            train = cls.loc[cls["fold"] >= 1, "score"]
            test = cls.loc[cls["fold"] == -1, "score"]
            ax.hist(train, bins=bins, density=True, histtype="stepfilled", alpha=0.3, color=color,
                    label=f"{name} train")
            counts, _ = np.histogram(test, bins=bins)
            norm = counts.sum() * np.diff(bins)
            p = ks[(b, name)]["p_value"]
            ax.errorbar(centres, counts / norm, yerr=np.sqrt(counts) / norm, fmt="o", ms=3, color=color,
                        label=f"{name} test (KS p = {p:.2g})")
        ax.set(xlabel="score", ylabel="normalised counts", yscale="log", title=_bin_label(edges, b))
        ax.legend(fontsize="small")
    return fig


def plot_efficiency_vs_threshold(df, doc):
    """Signal and background efficiency on the test set vs score threshold, with working points from fold 0."""
    edges = doc["pt_bins"]
    n = len(edges) - 1
    fig = Figure(figsize=(5 * n, 4.5), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, n))
    test = df[df["fold"] == -1]
    for b, ax in enumerate(axes):
        group = test[test["pt_bin"] == b]
        # roc_curve: thr decreasing, tpr/fpr = efficiency of score >= thr; thr[0] is an artificial +inf.
        fpr, tpr, thr = roc_curve(group["label"], group["score"])
        ax.plot(thr[1:], tpr[1:], color="tab:blue", label="signal")
        ax.plot(thr[1:], fpr[1:], color="tab:red", label="background")
        wps = [row for row in doc["working_points_test"] if row["pt_bin"] == b]
        for row, ls in zip(wps, ("--", "-.", ":", (0, (5, 5)))):
            ax.axvline(row["threshold"], ls=ls, color="black", lw=0.8,
                       label=f"{row['target']:.0%} on fold 0: test sig {row['eff_signal']:.1%}, bkg {row['eff_bkg']:.2%}")
        # Show only the range where something happens: from 99.5% signal efficiency to the highest score.
        lo = thr[min(np.searchsorted(tpr, 0.995), len(thr) - 1)]
        ax.set(xlabel="score threshold", ylabel="efficiency (test set)", yscale="log", title=_bin_label(edges, b),
               xlim=(lo, thr[1]))
        ax.legend(fontsize="x-small", loc="lower left")
    return fig


PLOTS = {
    "roc": plot_roc,
    "score_train_test": plot_score_train_test,
    "efficiency_vs_threshold": plot_efficiency_vs_threshold,
}


def write_plots(df, doc, out_dir):
    """Draw all validation plots into out_dir as PNG; return the list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, draw in PLOTS.items():
        path = out_dir / f"{name}.png"
        draw(df, doc).savefig(path, dpi=120)
        paths.append(path)
    return paths
