#!/usr/bin/env python
"""
Visualize the 3 self-context arms against each other.

Arms (as named in results_self_context/<dataset>.tsv, the `arm` column):
    random   -> no constraint on whether a sample appears in its own context
    exclude  -> context strictly excludes the sample (leave-one-out)
    include  -> context always contains the sample (label leak)

Output:
    - a table (mean +/- sample std over repeats) to stdout
    - one grouped bar chart per dataset (x = n, grouped by arm, std error bars),
      annotated with the delta vs the `random` baseline.
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
    "random": "random",
    "exclude": "exclude (LOO)",
    "include": "include (leak)",
}
ARM_ORDER = ["random", "exclude", "include"]
ARM_COLORS = {
    "random": "#9e9e9e",
    "exclude": "#42a5f5",
    "include": "#e53935",
}


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
            rows.setdefault((dataset, int(n), arm), []).append((train, val, test))


def load(path):
    """Return rows keyed (dataset, int(n), arm) -> list of (train, val, test)."""
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
    """Aggregate over repeats -> summary[(dataset, n)][arm] = dict of means/stds."""
    summary = {}
    for (dataset, n, arm), vals in rows.items():
        arms = summary.setdefault((dataset, n), {}).setdefault(arm, {"train": [], "val": [], "test": []})
        for v in vals:
            arms["train"].append(v[0])
            arms["val"].append(v[1])
            arms["test"].append(v[2])

    for key, arms in summary.items():
        for arm, m in arms.items():
            m["n_rep"] = len(m["test"])
            for metric in ("train", "val", "test"):
                mean, std = meanstd(m[metric])
                m[metric + "_mean"] = mean
                m[metric + "_std"] = std
    return summary


def print_table(summary):
    arm_rank = {a: i for i, a in enumerate(ARM_ORDER)}
    hdr = f"{'dataset':<22}{'n':>5}  {'arm':<18}{'rep':>4}  {'train':>18}  {'val':>18}  {'test':>18}"
    print(hdr)
    print("-" * len(hdr))
    flat = []
    for (dataset, n), arms in summary.items():
        for arm in arms:
            flat.append((dataset, n, arm_rank.get(arm, 99), arm, arms[arm]))
    flat.sort(key=lambda k: (k[0], k[1], k[2]))
    for dataset, n, _, arm, m in flat:
        print(f"{dataset:<22}{n:>5}  {ARM_DISPLAY.get(arm, arm):<18}{m['n_rep']:>4}  "
              f"{m['train_mean']:>9.4f}+-{m['train_std']:<7.4f}  "
              f"{m['val_mean']:>9.4f}+-{m['val_std']:<7.4f}  "
              f"{m['test_mean']:>9.4f}+-{m['test_std']:<7.4f}")
    print()


def draw_dataset(summary, dataset, ns, arms, metric, out_dir):
    metric_mean = metric + "_mean"
    metric_std = metric + "_std"

    x = np.arange(len(ns))
    width = 0.8 / len(arms)

    random_means = [100 * summary[(dataset, n)].get("random", {metric_mean: np.nan})[metric_mean] for n in ns]

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
    ax.legend(title="self_context", bbox_to_anchor=(1.02, 1), loc="upper left")

    # ---- annotate delta vs random for the non-baseline arms ----
    for a_i, arm in enumerate(arms):
        if arm == "random":
            continue
        offset = (a_i - (len(arms) - 1) / 2) * width
        for j, n in enumerate(ns):
            m = 100 * summary[(dataset, n)][arm][metric_mean]
            delta = m - random_means[j]
            color = "#2e7d32" if delta >= 0 else "#c62828"
            ax.annotate(f"{delta:+.2f}",
                        xy=(x[j] + offset, m),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=color)

    all_means = [100 * summary[(dataset, n)][arm][metric_mean] for n in ns for arm in arms]
    all_stds = [100 * summary[(dataset, n)][arm][metric_std] for n in ns for arm in arms]
    lo = min(m - s for m, s in zip(all_means, all_stds))
    hi = max(m + s for m, s in zip(all_means, all_stds))
    span = hi - lo
    if span <= 0:
        span = abs(hi) or 1.0
    pad = span * 0.15
    ax.set_ylim(lo - pad, hi + pad)

    fig.tight_layout()
    out = os.path.join(out_dir, f"self_context_{dataset}_{metric}.png")
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
    print_table(summary)

    metrics = [m for m in (["val", "test"] if args.metric == "both" else [args.metric])]
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = sorted({k[0] for k in summary})
    if args.dataset:
        datasets = [d for d in datasets if d == args.dataset]

    for dataset in datasets:
        ns = sorted({k[1] for k in summary if k[0] == dataset})
        arms = [a for a in ARM_ORDER if any(a in summary[(dataset, n)] for n in ns)]
        for metric in metrics:
            draw_dataset(summary, dataset, ns, arms, metric, args.out_dir)


if __name__ == "__main__":
    main()