#!/usr/bin/env bash
# Visualize the 2 test-time-augmentation strategies (tta_subset / tta_full).
# Usage: bash analysis/plot_tta.sh [tsv_or_dir] [metric]   (metric: val | test | both)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TSV="${1:-$REPO_ROOT/results_tta}"
METRIC="${2:-both}"

conda run -n tabmda python "$SCRIPT_DIR/plot_tta.py" \
    --tsv "$TSV" --metric "$METRIC"