#!/usr/bin/env bash
# Run the contrastive extension and the reference TabMDA pipeline under the
# same dataset / sample-size / repeat config, so the two can be compared.

# ---------------------------------------------------------------------------
#  Shared config
# ---------------------------------------------------------------------------
DATASET="${DATASET:-vehicle}"
NUM_REAL_SAMPLES="${NUM_REAL_SAMPLES:-100}"
REPEAT_ID="${REPEAT_ID:-0}"

# Number of contexts per sample (K in the contrastive module = num_contexts in TabMDA).
K="${K:-5}"

# Training epochs (epochs in the contrastive module = train_epochs in TabMDA).
EPOCHS="${EPOCHS:-30}"

# TabMDA-specific
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LogReg}"
CONTEXT_SIZE="${CONTEXT_SIZE:-0.5}"

# ---------------------------------------------------------------------------
#  1. Contrastive learning
# ---------------------------------------------------------------------------
python -m tabmda_contrastive.run train \
    --dataset "$DATASET" \
    --num_real_samples "$NUM_REAL_SAMPLES" \
    --repeat_id "$REPEAT_ID" \
    --K "$K" \
    --epochs "$EPOCHS" \
    --cache_dir results_contrastive/cache

echo "contrastive learning has done!"

# ---------------------------------------------------------------------------
#  2. TabMDA (reference)
# ---------------------------------------------------------------------------
python train.py \
    --dataset "$DATASET" \
    --num_real_samples "$NUM_REAL_SAMPLES" \
    --repeat_id "$REPEAT_ID" \
    --augmentor_model tabmda_encoder \
    --classifier_model "$CLASSIFIER_MODEL" \
    --num_contexts "$K" \
    --context_size "$CONTEXT_SIZE" \
    --train_epochs "$EPOCHS"

echo "TabMDA has done!"