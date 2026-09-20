#!/usr/bin/env bash
# Draw heatmaps for every (dataset, n) in the results TSV directory.
# Usage: bash analysis/plot_all_heatmaps.sh [tsv_or_dir] [metric]   (metric: val | test | both)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TSV="${1:-$REPO_ROOT/results}"
METRIC="${2:-both}"

if [ -d "$TSV" ]; then
    # Directory of per-dataset *.tsv files (run_comparison.sh output): merge them.
    CAT_ARGS=( $(printf '%s\n' "$TSV"/*.tsv | sort) )
    if [ "${#CAT_ARGS[@]}" -eq 0 ]; then
        echo "no *.tsv files in $TSV" >&2
        exit 1
    fi
    awk -F'\t' -v OFS='\t' \
        'FNR > 1 && NF >= 5 && $3 == "tabmda_encoder" {print $1, $2}' "${CAT_ARGS[@]}" \
        | sort -u \
        | while IFS=$'\t' read -r dataset n; do
              [ -z "$dataset" ] && continue
              echo "===== dataset=${dataset} n=${n} ====="
              conda run -n tabmda python "$SCRIPT_DIR/plot_heatmap.py" \
                  --tsv "$TSV" --dataset "$dataset" --n "$n" --metric "$METRIC"
          done
else
    awk -F'\t' -v OFS='\t' \
        'NR > 1 && NF >= 5 && $3 == "tabmda_encoder" {print $1, $2}' "$TSV" \
        | sort -u \
        | while IFS=$'\t' read -r dataset n; do
              [ -z "$dataset" ] && continue
              echo "===== dataset=${dataset} n=${n} ====="
              conda run -n tabmda python "$SCRIPT_DIR/plot_heatmap.py" \
                  --tsv "$TSV" --dataset "$dataset" --n "$n" --metric "$METRIC"
          done
fi