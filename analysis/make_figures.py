"""Rebuild every table and figure of the paper from the per-batch results.

    python analysis/make_figures.py                       # published results
    python analysis/make_figures.py --input my_run.csv    # your own run

Outputs go to ``results/summary`` (CSV / Markdown) and ``results/figures``
(PNG + SVG).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import cliques, friedman, nemenyi_cd, rank_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "results", "raw", "exp040_per_batch_mae.csv.gz")

# The 13 methods compared in the paper (11 baselines + 2 SMQ variants)
METHODS = ["SMQ_ISO", "SMQ", "CC", "PCC", "AC", "PAC", "EMQ", "EMQ_BCTS",
           "KDEyML", "KDEyCS", "KDEyHD", "FM", "PWK"]
LABELS = {"SMQ_ISO": "SMQ-ISO", "EMQ_BCTS": "EMQ-BCTS", "KDEyML": "KDEy-ML",
          "KDEyCS": "KDEy-CS", "KDEyHD": "KDEy-HD"}
OURS = {"SMQ", "SMQ_ISO"}

ACCENT = "#B03A48"   # SMQ (same crimson as the slides)
INK = "#1F2937"
MUTED = "#9CA3AF"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def label(m):
    return LABELS.get(m, m)


# ------------------------------------------------------------------ data
def load(path):
    """Per-batch results of either the original script or run_benchmark.py."""
    df = pd.read_csv(path)
    df = df.rename(columns={"modelo": "method", "erro": "mae", "n_classes_original": "n_classes"})
    df["method"] = (
        df["method"]
        .str.replace(r"^MoSS_BASE_\d+$", "SMQ", regex=True)
        .str.replace(r"^MoSS_ISO_CALIBRATED_\d+$", "SMQ_ISO", regex=True)
        .str.replace(r"^MoSS_RF_CALIBRATED_\d+$", "SMQ_RF", regex=True)
        .replace({"EMQ_MLQ": "EMQ", "EMQ_BCTS_MLQ": "EMQ_BCTS"})
    )
    return df


def mae_table(df):
    """Mean MAE per dataset (rows) and method (columns)."""
    table = df[df["method"].isin(METHODS)].pivot_table(
        index="dataset", columns="method", values="mae", aggfunc="mean"
    )[METHODS]
    n_classes = df.groupby("dataset")["n_classes"].first()
    incomplete = table.index[table.isna().any(axis=1)].tolist()
    return table.drop(index=incomplete), n_classes, incomplete


# --------------------------------------------------------------- figures
def save(fig, name, out_dir):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)


def cd_diagram(avg, cd, title, out_dir, name):
    """Critical-difference diagram (Demšar, 2006)."""
    avg = avg.sort_values()
    k = len(avg)
    lo, hi = 1, int(np.ceil(avg.max()))
    half = int(np.ceil(k / 2))
    left, right = avg.iloc[:half], avg.iloc[half:][::-1]
    groups = cliques(avg, cd)

    fig_h = 1.6 + 0.28 * max(len(left), len(right)) + 0.18 * len(groups)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_xlim(lo - 1.9, hi + 1.9)
    top = 0.0
    ax.set_ylim(-(0.45 + 0.28 * max(len(left), len(right)) + 0.18 * len(groups)), 1.05)
    ax.axis("off")

    # axis with ticks
    ax.plot([lo, hi], [top, top], color=INK, lw=1.2)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [top, top + 0.08], color=INK, lw=1)
        ax.text(t, top + 0.16, str(t), ha="center", va="bottom", fontsize=9, color=INK)

    # CD bar
    ax.plot([lo, lo + cd], [0.78, 0.78], color=INK, lw=1.2)
    for x in (lo, lo + cd):
        ax.plot([x, x], [0.74, 0.82], color=INK, lw=1.2)
    ax.text(lo + cd / 2, 0.86, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9, color=INK)

    # clique bars (methods not significantly different)
    y_bar = -0.14
    for g in groups:
        ax.plot([avg[g[0]] - 0.03, avg[g[-1]] + 0.03], [y_bar, y_bar], color=MUTED, lw=4,
                solid_capstyle="round")
        y_bar -= 0.18
    base = y_bar - 0.05

    def draw(side, names, xs):
        for i, (m, r) in enumerate(zip(names, xs)):
            y = base - 0.28 * i
            color = ACCENT if m in OURS else INK
            weight = "bold" if m in OURS else "normal"
            end = lo - 0.25 if side == "left" else hi + 0.25
            ax.plot([r, r, end], [top, y, y], color=color, lw=1.4 if m in OURS else 0.9)
            txt = f"{label(m)}  ({r:.2f})" if side == "left" else f"({r:.2f})  {label(m)}"
            ax.text(end - 0.05 if side == "left" else end + 0.05, y, txt,
                    ha="right" if side == "left" else "left", va="center",
                    color=color, fontsize=9.5, fontweight=weight)

    draw("left", list(left.index), list(left.values))
    draw("right", list(right.index), list(right.values))
    ax.set_title(title, fontsize=11, color=INK, pad=6, loc="left")
    save(fig, name, out_dir)


def rank_boxplot(ranks, title, out_dir, name):
    """Distribution of per-dataset ranks, ordered by median (paper Figs. 2-3)."""
    order = ranks.median().sort_values(kind="stable").index.tolist()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    data = [ranks[m].values for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                    medianprops={"color": INK, "lw": 1.6},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                                "markeredgecolor": MUTED})
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(ACCENT if m in OURS else "#D1D5DB")
        patch.set_alpha(0.9 if m in OURS else 1.0)
        patch.set_edgecolor(INK)
    for whisker in bp["whiskers"] + bp["caps"]:
        whisker.set_color(MUTED)
    ax.set_xticks(range(1, len(order) + 1), [label(m) for m in order], rotation=35, ha="right")
    for tick, m in zip(ax.get_xticklabels(), order):
        if m in OURS:
            tick.set_color(ACCENT)
            tick.set_fontweight("bold")
    ax.set_ylabel("Rank per dataset (1 = best)")
    ax.set_ylim(0.5, len(order) + 0.5)
    ax.invert_yaxis()
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    save(fig, name, out_dir)


def rank_vs_classes(mae, n_classes, out_dir, name):
    """Average rank of each method on datasets with at least k classes."""
    thresholds = sorted(n_classes.loc[mae.index].unique())
    thresholds = [t for t in thresholds if (n_classes.loc[mae.index] >= t).sum() >= 6]
    curves, counts = {m: [] for m in METHODS}, []
    for t in thresholds:
        sub = mae[n_classes.loc[mae.index] >= t]
        counts.append(len(sub))
        avg = rank_table(sub).mean()
        for m in METHODS:
            curves[m].append(avg[m])

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(thresholds))
    for m in METHODS:
        if m in OURS:
            continue
        ax.plot(x, curves[m], color="#CBD5E1", lw=1.2)
        ax.text(x[-1] + 0.1, curves[m][-1], label(m), fontsize=7.5, color=MUTED, va="center")
    for m, style in (("SMQ", "--"), ("SMQ_ISO", "-")):
        ax.plot(x, curves[m], color=ACCENT, lw=2.4, ls=style, marker="o", ms=4)
        ax.text(x[-1] + 0.1, curves[m][-1], label(m), fontsize=9, color=ACCENT,
                fontweight="bold", va="center")
    ax.set_xticks(x, [f"≥{t}\n({c})" for t, c in zip(thresholds, counts)], fontsize=8)
    ax.set_xlabel("Minimum number of classes (number of datasets)")
    ax.set_ylabel("Average rank (1 = best)")
    ax.invert_yaxis()
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.3, len(x) - 0.2 + 1.0)
    ax.set_title("Average rank as low-cardinality datasets are removed", fontsize=11,
                 color=INK, loc="left")
    save(fig, name, out_dir)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--min-classes", type=int, default=11,
                    help="threshold of the high-cardinality analysis")
    args = ap.parse_args(argv)

    fig_dir = os.path.join(args.out, "figures")
    sum_dir = os.path.join(args.out, "summary")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(sum_dir, exist_ok=True)

    df = load(args.input)
    mae, n_classes, incomplete = mae_table(df)
    if incomplete:
        print(f"excluded (some baseline failed, incomplete comparison): {incomplete}")

    table = mae.copy()
    table.insert(0, "n_classes", n_classes.loc[table.index])
    table.sort_values(["n_classes", "dataset"]).to_csv(
        os.path.join(sum_dir, "mae_by_dataset.csv"), float_format="%.6f")

    scopes = {
        "all": (mae, f"All datasets (3–{int(n_classes.max())} classes)"),
        "high": (mae[n_classes.loc[mae.index] >= args.min_classes],
                 f"Datasets with ≥ {args.min_classes} classes"),
    }
    report = {"excluded_datasets": incomplete}
    md = ["# Summary statistics", "",
          "Mean absolute error (MAE) per dataset is the average over its 1000 test batches.",
          "Ranks are computed per dataset (1 = lowest MAE) and averaged.", ""]

    for key, (sub, title) in scopes.items():
        ranks = rank_table(sub)
        avg = ranks.mean().sort_values()
        stats = friedman(sub)
        cd = nemenyi_cd(stats["n_methods"], stats["n_datasets"])
        best = avg.index[0]
        not_diff = [m for m in avg.index if avg[m] - avg[best] < cd]
        stats.update({"nemenyi_cd": cd, "best": best,
                      "not_significantly_different_from_best": not_diff})
        report[key] = stats

        summary = pd.DataFrame({
            "avg_rank": avg,
            "median_rank": ranks.median()[avg.index],
            "median_mae": sub.median()[avg.index],
            "mean_mae": sub.mean()[avg.index],
        })
        summary.index.name = "method"
        summary.to_csv(os.path.join(sum_dir, f"ranks_{key}.csv"), float_format="%.4f")

        n_low, n_high = int(n_classes.loc[sub.index].min()), int(n_classes.loc[sub.index].max())
        tag = f"{n_low}-{n_high}"
        cd_diagram(avg, cd, f"{title} — {stats['n_datasets']} datasets, Friedman p = "
                   f"{stats['iman_davenport_p']:.1e}", fig_dir, f"cd_diagram_{tag}")
        rank_boxplot(ranks, f"Per-dataset ranks, {title.lower()}", fig_dir, f"rank_boxplot_{tag}")

        md += [f"## {title}", "",
               f"{stats['n_datasets']} datasets, {stats['n_methods']} methods. "
               f"Friedman χ² = {stats['chi2']:.2f} (p = {stats['chi2_p']:.2e}); "
               f"Iman–Davenport F = {stats['iman_davenport_F']:.2f} "
               f"(p = {stats['iman_davenport_p']:.2e}); Nemenyi CD = {cd:.3f} (α = 0.05).", "",
               "| Method | Avg. rank | Median rank | Median MAE | Mean MAE |",
               "|---|---:|---:|---:|---:|"]
        for m, row in summary.iterrows():
            name = f"**{label(m)}**" if m in OURS else label(m)
            md.append(f"| {name} | {row.avg_rank:.2f} | {row.median_rank:.1f} | "
                      f"{row.median_mae:.4f} | {row.mean_mae:.4f} |")
        md += ["", f"Not significantly different from the best ({label(best)}): "
               + ", ".join(label(m) for m in not_diff) + ".", ""]
        print(f"[{key}] best={best} CD={cd:.3f}\n{avg.round(2).to_string()}\n")

    rank_vs_classes(mae, n_classes, fig_dir, "rank_vs_min_classes")
    with open(os.path.join(sum_dir, "statistics.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(sum_dir, "README.md"), "w") as f:
        f.write("\n".join(md))
    print(f"tables -> {sum_dir}\nfigures -> {fig_dir}")


if __name__ == "__main__":
    main()
