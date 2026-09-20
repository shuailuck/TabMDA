#!/usr/bin/env bash
# Draw heatmaps for every (dataset, n) in the results TSV.
# Usage: bash analysis/plot_all_heatmaps.sh [tsv] [metric]   (metric: val | test | both)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TSV="${1:-$REPO_ROOT/results/results.tsv}"
METRIC="${2:-both}"

awk -F'\t' -v OFS='\t' \
    'NR > 1 && NF >= 5 && $3 == "tabmda_encoder" {print $1, $2}' "$TSV" \
    | sort -u \
    | while IFS=$'\t' read -r dataset n; do
          [ -z "$dataset" ] && continue
          echo "===== dataset=${dataset} n=${n} ====="
          conda run -n tabmda python "$SCRIPT_DIR/plot_heatmap.py" \
              --dataset "$dataset" --n "$n" --metric "$METRIC"
      done