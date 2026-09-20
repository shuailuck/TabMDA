#!/usr/bin/env bash
# Visualize the 4 augmentation arms from results.tsv.
#
# Arms: Real (none), SMOTE, Embedding (tabmda_embedding), Encoder (tabmda_encoder).
# The Encoder arm is grid-searched; this script selects the best grid point per
# (dataset, n) by mean VALIDATION balanced accuracy (replicates run_comparison.sh).
#
# Usage:
#   bash analysis/plot_comparison.sh                          # all datasets, val + test
#   bash analysis/plot_comparison.sh test                     # all datasets, test only
#   bash analysis/plot_comparison.sh both qsar-biodeg         # one dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METRIC="${1:-both}"
DATASET_ARG="${2:-}"

ARGS=(--metric "$METRIC")
if [ -n "$DATASET_ARG" ]; then
    ARGS+=(--dataset "$DATASET_ARG")
fi

conda run -n tabmda python "$SCRIPT_DIR/plot_comparison.py" "${ARGS[@]}"