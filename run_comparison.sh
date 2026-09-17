#!/usr/bin/env bash
# ============================================================================
#  TabMDA 4-way comparison batch runner
#
#  For each dataset (and each repeat), this script runs 4 experiments that
#  compare data augmentation in the INPUT space vs the EMBEDDING space:
#
#      arm                  space         augmentation
#    ------------          ----------     ---------------------------
#    1. none               input          original data, no augmentation
#    2. smote              input          SMOTE augmentation
#    3. tabmda_embedding   embedding      TabPFN encode once, no augmentation
#    4. tabmda_encoder     embedding      TabMDA (context subsetting)
#
#  Usage:
#      bash run_comparison.sh            # run everything with defaults
#      DRY_RUN=1 bash run_comparison.sh  # print commands only, run nothing
#
#  All settings below can be overridden via environment variables, e.g.:
#      DATASETS="vehicle:100 qsar-biodeg:20" REPEATS="0" bash run_comparison.sh
# ============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-tabmda}"          # conda env holding the TabPFN v1 install
PYTHON_BIN="${PYTHON_BIN:-}"              # override with a full path if needed

# ---------------------------------------------------------------------------
#  Global experiment settings
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LogReg}"

NUM_CONTEXTS="${NUM_CONTEXTS:-20}"        # contexts for tabmda_encoder (train)
CONTEXT_SIZE="${CONTEXT_SIZE:-0}"         # 0 => random 0.5..0.99 per context
NUM_CONTEXTS_VAL="${NUM_CONTEXTS_VAL:-1}"
NUM_CONTEXTS_TEST="${NUM_CONTEXTS_TEST:-1}"
AGGREGATION_TEST="${AGGREGATION_TEST:-mean}"

SMOTE_K="${SMOTE_K:-3}"                   # SMOTE k-nearest neighbours
SMOTE_ROUNDS="${SMOTE_ROUNDS:-1}"         # SMOTE rounds (each round ~doubles)

ENABLE_WANDB="${ENABLE_WANDB:-0}"         # 1 to enable, 0 to disable (needs api key)

REPEATS="${REPEATS:-0 1 2 3 4 5 6 7 8 9}" # repeat ids (0..9)

# ---------------------------------------------------------------------------
#  Per-dataset configuration
#
#  Format:  "dataset:num_real_samples"
#  Supported num_real_samples per dataset (see dataset/datasets.py):
#      qsar-biodeg         20 50 100 200 500 -1
#      vehicle             20 50 100 200 -1
#      texture             50 100 200 500 -1
#      steel-plates-fault  20 50 100 200 500 -1
#      MiceProtein         50 100 200 500 -1
#      mfeat-fourier       50 100 200 500 -1
# ---------------------------------------------------------------------------
DATASETS="${DATASETS:-qsar-biodeg:50 vehicle:50 texture:50 steel-plates-fault:50 MiceProtein:50 mfeat-fourier:50}"

# ---------------------------------------------------------------------------
#  The 4 comparison arms (order matters for readability of results)
# ---------------------------------------------------------------------------
ARMS="${ARMS:-none smote tabmda_embedding tabmda_encoder}"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
log()  { echo "[run_comparison] $*"; }
fail() { echo "[run_comparison] ERROR: $*" >&2; }

# Resolve the python command as an ARRAY (handles `conda run -n env python`).
resolve_python() {
    if [ -n "$PYTHON_BIN" ]; then
        PY_CMD=("$PYTHON_BIN")
    elif [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
        PY_CMD=(python)
    elif command -v conda >/dev/null 2>&1; then
        # NOTE: `conda run` buffers stdout/stderr until the child finishes.
        PY_CMD=(conda run -n "$CONDA_ENV" python)
    else
        PY_CMD=(python)
    fi
}

# Execute a command, honouring the wandb toggle.
run() {
    if [ "$ENABLE_WANDB" = "0" ]; then
        WANDB_MODE=disabled "$@"
    else
        "$@"
    fi
}

# Extra flags unique to each arm (as an array).
arm_args_for() {
    case "$1" in
        none)
            ARM_ARGS=(--augmentor_model none)
            ;;
        smote)
            ARM_ARGS=(--augmentor_model smote --smote_k "$SMOTE_K" --smote_rounds "$SMOTE_ROUNDS")
            ;;
        tabmda_embedding)
            ARM_ARGS=(--augmentor_model tabmda_embedding)
            ;;
        tabmda_encoder)
            ARM_ARGS=(--augmentor_model tabmda_encoder
                      --context_size "$CONTEXT_SIZE" --num_contexts "$NUM_CONTEXTS"
                      --num_contexts_val "$NUM_CONTEXTS_VAL" --num_contexts_test "$NUM_CONTEXTS_TEST"
                      --aggregation_over_contexts_test "$AGGREGATION_TEST")
            ;;
        *)
            fail "unknown arm: $1"
            ARM_ARGS=()
            return 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
resolve_python
log "python : ${PY_CMD[*]}"
log "classifier: $CLASSIFIER_MODEL  |  wandb: $ENABLE_WANDB  |  dry-run: $DRY_RUN"
log "datasets : $DATASETS"
log "repeats  : $REPEATS"
log "arms     : $ARMS"
echo

FAILED=0
TOTAL=0

for entry in $DATASETS; do
    dataset="${entry%%:*}"
    nrs="${entry##*:}"

    for repeat in $REPEATS; do
        for arm in $ARMS; do
            TOTAL=$((TOTAL + 1))

            if ! arm_args_for "$arm"; then
                continue
            fi

            log "== [$TOTAL] dataset=$dataset  n=$nrs  repeat=$repeat  arm=$arm =="

            if [ "$DRY_RUN" = "1" ]; then
                echo "   ${PY_CMD[*]} train.py --dataset \"$dataset\" --num_real_samples $nrs \
--repeat_id $repeat --classifier_model $CLASSIFIER_MODEL ${ARM_ARGS[*]}"
                continue
            fi

            run "${PY_CMD[@]}" train.py \
                --dataset "$dataset" \
                --num_real_samples "$nrs" \
                --repeat_id "$repeat" \
                --classifier_model "$CLASSIFIER_MODEL" \
                "${ARM_ARGS[@]}"

            rc=$?
            if [ "$rc" -ne 0 ]; then
                fail "run failed (exit $rc): dataset=$dataset n=$nrs repeat=$repeat arm=$arm"
                FAILED=$((FAILED + 1))
            fi
            echo
        done
    done
done

# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------
log "done. total=$TOTAL  failed=$FAILED"
if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
