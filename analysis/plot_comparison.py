#!/usr/bin/env python
"""
Visualize the 4 augmentation arms against each other.

Arms (as named in results.tsv):
    none              -> Real         (original data, no augmentation)
    smote             -> SMOTE        (SMOTE augmentation in input space)
    tabmda_embedding  -> Embedding    (TabPFN encode once, no subsetting)
    tabmda_encoder    -> Encoder      (TabMDA: in-context subsetting grid)

The `tabmda_encoder` arm is grid-searched over (context_size x num_contexts).
Following run_comparison.sh, we FIRST select the single best grid point per
(dataset, n) by mean VALIDATION balanced accuracy, then report only that point.

Output:
    - a table (mean +/- sample std over repeats) to stdout
    - one grouped bar chart per dataset (x = n, grouped by arm, std error bars)
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

ARM_DISPLAY = {
    "none": "Real",
    "smote": "SMOTE",
    "tabmda_embedding": "Embedding",
    "tabmda_encoder": "Encoder",
}
ARM_ORDER = ["none", "smote", "tabmda_embedding", "tabmda_encoder"]
ARM_COLORS = {
    "none": "#9e9e9e",
    "smote": "#42a5f5",
    "tabmda_embedding": "#ffa726",
    "tabmda_encoder": "#e53935",
}


def load_tsv(path, rows):
    """Append rows from a single TSV into `rows` (keyed like load())."""
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
    """Return rows keyed (dataset, int(n), arm, cs, nc) -> list of (train, val, test).

    `path` may be a single TSV file or a directory of `*.tsv` files (one per
    dataset, as produced by run_comparison.sh).
    """
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


def select_best_encoder(rows):
    """Replicate run_comparison.sh: best (cs, nc) per (dataset, n) by mean val accuracy."""
    chosen = {}  # (dataset, n) -> (val_mean, cs, nc)
    for (dataset, n, arm, cs, nc), vals in rows.items():
        if arm != "tabmda_encoder":
            continue
        vm = sum(v[1] for v in vals) / len(vals)
        key = (dataset, n)
        if key not in chosen or vm > chosen[key][0]:
            chosen[key] = (vm, cs, nc)
    return chosen


def meanstd(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return m, s


def aggregate(rows, chosen):
    """Aggregate over repeats -> summary[(dataset, n)][arm] = dict of means/stds.

    Only the selected grid point is kept for the encoder arm.
    """
    summary = {}
    for (dataset, n, arm, cs, nc), vals in rows.items():
        if arm == "tabmda_encoder":
            vm, c_chosen, n_chosen = chosen.get((dataset, n), (None, None, None))
            if (cs, nc) != (c_chosen, n_chosen):
                continue
        key = (dataset, n)
        summary.setdefault(key, {}).setdefault(arm, {"train": [], "val": [], "test": []})
        for v in vals:
            summary[key][arm]["train"].append(v[0])
            summary[key][arm]["val"].append(v[1])
            summary[key][arm]["test"].append(v[2])

    for key, arms in summary.items():
        for arm, m in arms.items():
            m["n_rep"] = len(m["test"])
            for metric in ("train", "val", "test"):
                mean, std = meanstd(m[metric])
                m[metric + "_mean"] = mean
                m[metric + "_std"] = std
    return summary


def print_table(summary, chosen):
    order = ARM_ORDER
    arm_rank = {a: i for i, a in enumerate(order)}
    hdr = f"{'dataset':<22}{'n':>5}  {'arm':<30}{'rep':>4}  {'train':>18}  {'val':>18}  {'test':>18}"
    print(hdr)
    print("-" * len(hdr))
    flat = []
    for (dataset, n), arms in summary.items():
        for arm in arms:
            flat.append((dataset, n, arm_rank.get(arm, 99), arm, arms[arm]))
    flat.sort(key=lambda k: (k[0], k[1], k[2]))
    for dataset, n, _, arm, m in flat:
        label = ARM_DISPLAY.get(arm, arm)
        if arm == "tabmda_encoder":
            vm, cs, nc = chosen[(dataset, n)]
            label = f"Encoder (cs={cs}, nc={nc})"
        print(f"{dataset:<22}{n:>5}  {label:<30}{m['n_rep']:>4}  "
              f"{m['train_mean']:>9.4f}+-{m['train_std']:<7.4f}  "
              f"{m['val_mean']:>9.4f}+-{m['val_std']:<7.4f}  "
              f"{m['test_mean']:>9.4f}+-{m['test_std']:<7.4f}")
    print()


def draw_dataset(summary, chosen, dataset, ns, arms, metric, out_dir):
    metric_mean = metric + "_mean"
    metric_std = metric + "_std"

    x = np.arange(len(ns))
    width = 0.8 / len(arms)

    # Real (none) is the baseline; other arms are shown as deltas relative to it.
    # Values are scaled x100 to percent so bars are readable (avoids long decimal zeros).
    real_means = [100 * summary[(dataset, n)]["none"][metric_mean] for n in ns]

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(ns)), 6))
    for a_i, arm in enumerate(arms):
        means = [100 * summary[(dataset, n)][arm][metric_mean] for n in ns]
        stds = [100 * summary[(dataset, n)][arm][metric_std] for n in ns]
        offset = (a_i - (len(arms) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=ARM_DISPLAY.get(arm, arm), color=ARM_COLORS.get(arm),
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel(f"{metric} balanced accuracy (%)")
    ax.set_title(f"{dataset} — {metric} balanced accuracy (mean +/- std over repeats)")
    ax.legend(title="arm", bbox_to_anchor=(1.02, 1), loc="upper left")

    # ---- annotate delta vs Real for the other 3 arms ----
    # (positive delta = better than Real, green; negative = worse, red)
    for a_i, arm in enumerate(arms):
        if arm == "none":
            continue
        offset = (a_i - (len(arms) - 1) / 2) * width
        for j, n in enumerate(ns):
            m = 100 * summary[(dataset, n)][arm][metric_mean]
            delta = m - real_means[j]
            color = "#2e7d32" if delta >= 0 else "#c62828"
            ax.annotate(f"{delta:+.2f}",
                        xy=(x[j] + offset, m),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=color)

    # ---- y-axis auto-scaled to the data (not anchored at 0) ----
    all_means = [100 * summary[(dataset, n)][arm][metric_mean] for n in ns for arm in arms]
    all_stds = [100 * summary[(dataset, n)][arm][metric_std] for n in ns for arm in arms]
    lo = min(m - s for m, s in zip(all_means, all_stds))
    hi = max(m + s for m, s in zip(all_means, all_stds))
    span = hi - lo
    if span <= 0:
        span = abs(hi) or 1.0
    pad = span * 0.15
    ax.set_ylim(lo - pad, hi + pad)

    # Annotate the chosen encoder grid point per n.
    subtitle = "  |  ".join(
        f"n={n}: cs={chosen[(dataset, n)][1]}, nc={chosen[(dataset, n)][2]}" for n in ns
    )
    ax.text(0.5, -0.10, f"Encoder selected by best mean val: {subtitle}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="gray")

    fig.tight_layout()
    out = os.path.join(out_dir, f"comparison_{dataset}_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=os.path.join(REPO_ROOT, "results"),
                    help="TSV file OR directory of per-dataset *.tsv files")
    ap.add_argument("--metric", choices=["val", "test", "both"], default="both",
                    help="which metric to plot")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "comparison"),
                    help="output directory for PNGs")
    ap.add_argument("--dataset", default=None, help="only plot this dataset")
    args = ap.parse_args()

    rows = load(args.tsv)
    if not rows:
        raise SystemExit("no rows found")

    chosen = select_best_encoder(rows)
    summary = aggregate(rows, chosen)

    print_table(summary, chosen)

    metrics = [m for m in (["val", "test"] if args.metric == "both" else [args.metric])]
    os.makedirs(args.out_dir, exist_ok=True)

    # group arms per (dataset, n) preserving ARM_ORDER for those present
    datasets = sorted({k[0] for k in summary})
    if args.dataset:
        datasets = [d for d in datasets if d == args.dataset]

    for dataset in datasets:
        ns = sorted({k[1] for k in summary if k[0] == dataset})
        arms = [a for a in ARM_ORDER if any(a in summary[(dataset, n)] for n in ns)]
        for metric in metrics:
            draw_dataset(summary, chosen, dataset, ns, arms, metric, args.out_dir)


if __name__ == "__main__":
    main()