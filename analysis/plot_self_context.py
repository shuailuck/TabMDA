#!/usr/bin/env python
"""
Visualize the 3 self-context arms with (context_size, num_contexts) kept separate.

Arms (the `arm` column in results_self_context/<dataset>.tsv):
    random   -> no constraint on whether a sample appears in its own context
    exclude  -> context strictly excludes the sample (leave-one-out)
    include  -> context always contains the sample (label leak)

Because each arm can be run under many (context_size x num_contexts) configs,
the results are NOT collapsed over configs. Output:
    - a table (mean +/- sample std over repeats), one row per (arm, config)
    - per (dataset, n, metric): a small-multiples heatmap figure
        top row    : one heatmap per arm (shared colour scale)
        bottom row : delta heatmaps exclude-random and include-random
                     (diverging colour scale centred at 0)
"""
import argparse
import glob
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from table_image import save_table_image

ARM_DISPLAY = {
    "random": "random",
    "exclude": "exclude (LOO)",
    "include": "include (leak)",
}
ARM_ORDER = ["random", "exclude", "include"]


def load_tsv(path, rows):
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            dataset, n, arm, cs, nc, repeat, train, val, test = p[:9]
            try:
                train, val, test = float(train), float(val), float(test)
            except ValueError:
                continue
            rows.setdefault((dataset, int(n), arm, cs, nc), []).append((train, val, test))


def load(path):
    """Return rows keyed (dataset, int(n), arm, cs, nc) -> list of (train, val, test)."""
    rows = {}
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.tsv")))
        if not files:
            raise SystemExit(f"no *.tsv files in {path}")
        for f in files:
            load_tsv(f, rows)
    else:
        load_tsv(path, rows)
    return rows


def meanstd(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return m, s


def aggregate(rows):
    """Aggregate over repeats -> summary[(dataset, n)][(arm, cs, nc)] = stats dict."""
    summary = {}
    for (dataset, n, arm, cs, nc), vals in rows.items():
        cell = summary.setdefault((dataset, n), {}).setdefault(
            (arm, cs, nc), {"train": [], "val": [], "test": []})
        for v in vals:
            cell["train"].append(v[0])
            cell["val"].append(v[1])
            cell["test"].append(v[2])

    for cells in summary.values():
        for m in cells.values():
            m["n_rep"] = len(m["test"])
            for metric in ("train", "val", "test"):
                mean, std = meanstd(m[metric])
                m[metric + "_mean"] = mean
                m[metric + "_std"] = std
    return summary


def print_table(summary):
    arm_rank = {a: i for i, a in enumerate(ARM_ORDER)}
    hdr = (f"{'dataset':<22}{'n':>5}  {'arm':<18}{'cs':>6}{'nc':>5}{'rep':>4}  "
           f"{'train':>18}  {'val':>18}  {'test':>18}")
    print(hdr)
    print("-" * len(hdr))
    flat = []
    for (dataset, n), cells in summary.items():
        for (arm, cs, nc), m in cells.items():
            flat.append((dataset, n, float(cs), int(nc), arm_rank.get(arm, 99), arm, cs, nc, m))
    flat.sort(key=lambda k: (k[0], k[1], k[2], k[3], k[4]))

    table_headers = ["dataset", "n", "arm", "context_size", "num_contexts", "rep", "train", "val", "test"]
    table_rows = []
    for dataset, n, _, _, _, arm, cs, nc, m in flat:
        print(f"{dataset:<22}{n:>5}  {ARM_DISPLAY.get(arm, arm):<18}{cs:>6}{nc:>5}{m['n_rep']:>4}  "
              f"{m['train_mean']:>9.4f}+-{m['train_std']:<7.4f}  "
              f"{m['val_mean']:>9.4f}+-{m['val_std']:<7.4f}  "
              f"{m['test_mean']:>9.4f}+-{m['test_std']:<7.4f}")
        table_rows.append([
            dataset, str(n), ARM_DISPLAY.get(arm, arm), cs, nc, str(m["n_rep"]),
            f"{m['train_mean']:.4f}+-{m['train_std']:.4f}",
            f"{m['val_mean']:.4f}+-{m['val_std']:.4f}",
            f"{m['test_mean']:.4f}+-{m['test_std']:.4f}",
        ])
    print()
    return table_headers, table_rows


def config_orders(cells):
    """Return (arm_order, cs_order, nc_order) for a (dataset, n) cell dict."""
    arms, cs_set, nc_set = set(), set(), set()
    for arm, cs, nc in cells:
        arms.add(arm)
        cs_set.add(cs)
        nc_set.add(nc)
    arm_order = [a for a in ARM_ORDER if a in arms]
    cs_order = sorted(cs_set, key=float)
    nc_order = sorted(nc_set, key=int)
    return arm_order, cs_order, nc_order


def cell_matrix(cells, arm, cs_order, nc_order, metric):
    key = metric + "_mean"
    M = np.full((len(cs_order), len(nc_order)), np.nan)
    for i, cs in enumerate(cs_order):
        for j, nc in enumerate(nc_order):
            m = cells.get((arm, cs, nc))
            if m is not None:
                M[i, j] = m[key]
    return M


def draw_heatmap(ax, M, cs_order, nc_order, title, vmin, vmax, cmap):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(nc_order)))
    ax.set_xticklabels(nc_order)
    ax.set_yticks(range(len(cs_order)))
    ax.set_yticklabels(cs_order)
    ax.set_xlabel("num_contexts")
    ax.set_ylabel("context_size")
    ax.set_title(title, fontsize=9)
    mid = (vmin + vmax) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < mid else "black")
    return im


def draw_config_grid(cells, dataset, n, metric, out_dir):
    arm_order, cs_order, nc_order = config_orders(cells)

    # ---- absolute accuracy heatmaps (one per arm, shared colour scale) ----
    matrices = {arm: cell_matrix(cells, arm, cs_order, nc_order, metric) for arm in arm_order}
    vmin = min(np.nanmin(M) for M in matrices.values())
    vmax = max(np.nanmax(M) for M in matrices.values())
    if np.isnan(vmin) or np.isnan(vmax):
        print(f"skip {dataset} n={n} {metric}: no data")
        return
    if vmin == vmax:
        vmax = vmin + 1e-9

    n_arms = len(arm_order)
    fig = plt.figure(figsize=(max(3.2 * n_arms, 6), 8))
    gs = fig.add_gridspec(2, n_arms, height_ratios=[1, 1], hspace=0.55, wspace=0.35)
    ax_arms = [fig.add_subplot(gs[0, i]) for i in range(n_arms)]
    ax_deltas = [fig.add_subplot(gs[1, i]) for i in range(n_arms)]

    im_abs = None
    for ax, arm in zip(ax_arms, arm_order):
        im_abs = draw_heatmap(ax, matrices[arm], cs_order, nc_order,
                              ARM_DISPLAY.get(arm, arm), vmin, vmax, "viridis")
    if im_abs is not None:
        fig.colorbar(im_abs, ax=ax_arms, fraction=0.03, pad=0.02, label=f"{metric} balanced accuracy")

    # ---- delta heatmaps (exclude-random, include-random), diverging, centred at 0 ----
    deltas = {}
    if "random" in matrices:
        for arm in ("exclude", "include"):
            if arm in matrices:
                deltas[arm] = matrices[arm] - matrices["random"]
    dmax = max(np.nanmax(np.abs(D)) for D in deltas.values()) if deltas else 0.0
    if dmax == 0:
        dmax = 1e-9

    im_delta = None
    for ax, arm in zip(ax_deltas, ("exclude", "include")):
        D = deltas.get(arm)
        if D is None:
            ax.axis("off")
            continue
        im_delta = draw_heatmap(ax, D, cs_order, nc_order, f"{ARM_DISPLAY[arm]} - random",
                                -dmax, dmax, "RdBu_r")
    if im_delta is not None and deltas:
        fig.colorbar(im_delta, ax=ax_deltas, fraction=0.03, pad=0.02, label=f"delta {metric} balanced accuracy")

    fig.suptitle(f"{dataset} n={n} — {metric} balanced accuracy (mean over repeats)", fontsize=10)
    out = os.path.join(out_dir, f"self_context_{dataset}_n{n}_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join(REPO_ROOT, "results_self_context"),
                    help="TSV file OR directory of per-dataset *.tsv files")
    ap.add_argument("--metric", choices=["val", "test", "both"], default="both",
                    help="which metric to plot")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results_self_context", "comparison"),
                    help="output directory for PNGs")
    ap.add_argument("--dataset", default=None, help="only plot this dataset")
    args = ap.parse_args()

    rows = load(args.tsv)
    if not rows:
        raise SystemExit("no rows found")

    summary = aggregate(rows)
    table_headers, table_rows = print_table(summary)
    save_table_image(
        table_headers, table_rows,
        title="Self-context comparison per (context_size, num_contexts) (mean +/- std over repeats)",
        path=os.path.join(args.out_dir, "table.png"),
    )

    metrics = [m for m in (["val", "test"] if args.metric == "both" else [args.metric])]
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = sorted({k[0] for k in summary})
    if args.dataset:
        datasets = [d for d in datasets if d == args.dataset]

    for dataset in datasets:
        ns = sorted({k[1] for k in summary if k[0] == dataset})
        for n in ns:
            cells = summary[(dataset, n)]
            for metric in metrics:
                draw_config_grid(cells, dataset, n, metric, args.out_dir)


if __name__ == "__main__":
    main()
