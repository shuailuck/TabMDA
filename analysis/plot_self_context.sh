#!/usr/bin/env bash
# Visualize the 3 self-context arms (random / exclude / include).
# Usage: bash analysis/plot_self_context.sh [tsv_or_dir] [metric]   (metric: val | test | both)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TSV="${1:-$REPO_ROOT/results_self_context}"
METRIC="${2:-both}"

conda run -n tabmda python "$SCRIPT_DIR/plot_self_context.py" \
    --tsv "$TSV" --metric "$METRIC"