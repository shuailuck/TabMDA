#!/usr/bin/env python
"""Heatmap of tabmda_encoder test/val balanced accuracy over context_size x num_contexts."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    rows = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            dataset, n, arm, cs, nc, repeat, train, val, test = p[:9]
            if arm != "tabmda_encoder":
                continue
            try:
                train, val, test = float(train), float(val), float(test)
            except ValueError:
                continue
            rows.setdefault((dataset, int(n), cs, nc), []).append((train, val, test))
    return rows


def grid(rows, metric_idx):
    cs_order = sorted({k[2] for k in rows}, key=float)
    nc_order = sorted({k[3] for k in rows}, key=int)
    M = np.zeros((len(cs_order), len(nc_order)))
    for i, cs in enumerate(cs_order):
        for j, nc in enumerate(nc_order):
            vals = rows.get((dataset, n, cs, nc))
            if vals:
                M[i, j] = np.mean([v[metric_idx] for v in vals])
            else:
                M[i, j] = np.nan
    return M, cs_order, nc_order


def draw(M, cs_order, nc_order, title, path, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(M)
    if vmax is None:
        vmax = np.nanmax(M)
    fig, ax = plt.subplots(figsize=(len(nc_order) * 1.6 + 2, len(cs_order) * 1.6 + 1.5))
    im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(nc_order)))
    ax.set_xticklabels(nc_order)
    ax.set_yticks(range(len(cs_order)))
    ax.set_yticklabels(cs_order)
    ax.set_xlabel("num_contexts")
    ax.set_ylabel("context_size")
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                        color="white" if M[i, j] < (vmin + vmax) / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved: {path}")


def print_grid(rows, dataset, n):
    cs_order = sorted({k[2] for k in rows}, key=float)
    nc_order = sorted({k[3] for k in rows}, key=int)
    hdr = ["context_size"] + [f"nc={nc}" for nc in nc_order]
    print("\t".join(hdr))
    for metric_name, idx in [("train", 0), ("val", 1), ("test", 2)]:
        print(f"--- {metric_name} balanced accuracy ---")
        for cs in cs_order:
            row = [str(cs)]
            for nc in nc_order:
                vals = rows.get((dataset, n, cs, nc))
                if vals:
                    row.append(f"{np.mean([v[idx] for v in vals]):.4f}")
                else:
                    row.append("nan")
            print("\t".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join(REPO_ROOT, "results", "results.tsv"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--print-grid", action="store_true", help="also print the mean-value table")
    ap.add_argument("--metric", choices=["val", "test", "both"], default="both",
                    help="which metric's heatmap to draw")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "heatmap"),
                    help="output directory for PNGs")
    args = ap.parse_args()

    rows = load(args.tsv)
    if not rows:
        raise SystemExit("no tabmda_encoder rows found")

    # If not specified, take the (dataset, n) that appears most often
    groups = {}
    for (d, n, cs, nc) in rows:
        groups[(d, n)] = groups.get((d, n), 0) + 1
    dataset, n = (args.dataset, args.n) if args.dataset and args.n else max(groups, key=groups.get)
    print(f"dataset={dataset} n={n}  ({len(rows)} cells, {groups[(dataset, n)]} config rows)")

    if args.print_grid:
        print_grid(rows, dataset, n)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    metric_map = {"val": (1, "val"), "test": (2, "test")}
    metrics = [metric_map[m] for m in (["val", "test"] if args.metric == "both" else [args.metric])]

    for idx, name in metrics:
        M, cs_order, nc_order = grid(rows, idx)
        draw(M, cs_order, nc_order,
             f"tabmda_encoder {name} balanced accuracy\n{dataset} n={n} (mean over repeats)",
             os.path.join(out_dir, f"heatmap_{dataset}_n{n}_{name}.png"))
