#!/usr/bin/env bash
# ============================================================================
#  Label-invariant "Top-K by TabPFN confidence" vs. TabMDA comparison runner
#
#  Runs two things under the SAME dataset / sample-size / repeat config:
#
#    1. experiment_label_invariant.py
#       - Samples a grid of contexts per query:  num_context x context_size.
#       - For each context C, records P(TabPFN(x_i | C) = y_i) alongside the
#         frozen-encoder embedding of x_i conditioned on C.
#       - Evaluates three downstream arms (LogReg on the TabPFN embeddings):
#           * per_cell            : every (num_contexts, context_size) cell
#           * top_k_by_probability: the Top-K contexts ranked by that probability
#           * tabmda_reference    : per_cell at context_size = TABMDA_CONTEXT_SIZE
#       - Writes the JSON to results_label_invariant/.
#
#    2. train.py --augmentor_model tabmda_encoder
#       - The reference TabMDA pipeline (construct_tabmda_model -> LogReg).
#       - Its test balanced-accuracy is the "TabMDA scheme" anchor that the
#         top_k_by_probability arm should beat.
#
#  Usage:
#      bash run_label_invariant_experiment.sh            # defaults (vehicle@100)
#      DRY_RUN=1 bash run_label_invariant_experiment.sh  # print commands only
#      MAX_SAMPLES=20 bash run_label_invariant_experiment.sh  # quick trend first
#
#  All settings can be overridden via environment variables, e.g.:
#      DATASET=texture NUM_REAL_SAMPLES=50 REPEAT_ID=1 bash run_label_invariant_experiment.sh
#      GRID_NUM_CONTEXTS="5,20" GRID_CONTEXT_SIZE="0.5,0.9" TOP_K="5,20" bash run_label_invariant_experiment.sh
# ============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-tabmda}"          # conda env holding the TabPFN v1 install
PYTHON_BIN="${PYTHON_BIN:-}"              # override with a full path if needed

# ---------------------------------------------------------------------------
#  Shared config
# ---------------------------------------------------------------------------
DATASET="${DATASET:-vehicle}"
NUM_REAL_SAMPLES="${NUM_REAL_SAMPLES:-100}"
REPEAT_ID="${REPEAT_ID:-0}"

# ---------------------------------------------------------------------------
#  Label-invariant experiment settings
# ---------------------------------------------------------------------------
GRID_NUM_CONTEXTS="${GRID_NUM_CONTEXTS:-5,20,50}"   # comma-separated context counts
GRID_CONTEXT_SIZE="${GRID_CONTEXT_SIZE:-0.5,0.7,0.9}"  # comma-separated proportions
TOP_K="${TOP_K:-5,20,50}"                           # comma-separated Top-K values
TABMDA_CONTEXT_SIZE="${TABMDA_CONTEXT_SIZE:-0.5}"     # cs of the internal TabMDA reference arm
SELF_CONTEXT="${SELF_CONTEXT:-random}"                # "random" matches TabMDA default
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"                       # e.g. 20; empty = full train set

# ---------------------------------------------------------------------------
#  TabMDA reference settings (train.py)
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LogReg}"
CONTEXT_SIZE="${CONTEXT_SIZE:-0.5}"                  # should match TABMDA_CONTEXT_SIZE
NUM_CONTEXTS="${NUM_CONTEXTS:-5}"                    # should match the lowest TOP_K
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"

# ---------------------------------------------------------------------------
#  Output
# ---------------------------------------------------------------------------
ENABLE_WANDB="${ENABLE_WANDB:-0}"                    # 1 to enable, 0 to disable
OUT_DIR="${OUT_DIR:-results_label_invariant}"
LOGS_DIR="$OUT_DIR/logs"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
log()  { echo "[run_label_invariant] $*"; }
fail() { echo "[run_label_invariant] ERROR: $*" >&2; }

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

# ---------------------------------------------------------------------------
#  Resolve python and print the plan
# ---------------------------------------------------------------------------
resolve_python
log "python           : ${PY_CMD[*]}"
log "dataset          : $DATASET  n=$NUM_REAL_SAMPLES  repeat=$REPEAT_ID  wandb=$ENABLE_WANDB"
log "label-inv grid   : num_contexts=[$GRID_NUM_CONTEXTS]  context_size=[$GRID_CONTEXT_SIZE]  top_k=[$TOP_K]"
log "label-inv self   : self_context=$SELF_CONTEXT  seed=$SEED  max_samples=${MAX_SAMPLES:-full}"
log "tabmda reference : num_contexts=$NUM_CONTEXTS  context_size=$CONTEXT_SIZE  classifier=$CLASSIFIER_MODEL"
if [ "$DRY_RUN" = "1" ]; then
    log "dry-run: commands below are NOT executed."
fi
echo

mkdir -p "$LOGS_DIR" "$OUT_DIR"

# ---------------------------------------------------------------------------
#  1. Label-invariant experiment (internal: per_cell / top_k / tabmda_reference)
# ---------------------------------------------------------------------------
LV_EXTRA=()
[ -n "$MAX_SAMPLES" ] && LV_EXTRA+=(--max_samples "$MAX_SAMPLES")

LV_CMD=(
    "${PY_CMD[@]}"
    experiment_label_invariant.py
    --dataset "$DATASET"
    --num_real_samples "$NUM_REAL_SAMPLES"
    --repeat_id "$REPEAT_ID"
    --grid_num_contexts "$GRID_NUM_CONTEXTS"
    --grid_context_size "$GRID_CONTEXT_SIZE"
    --top_k "$TOP_K"
    --tabmda_context_size "$TABMDA_CONTEXT_SIZE"
    --self_context "$SELF_CONTEXT"
    --seed "$SEED"
    "${LV_EXTRA[@]}"
    --out_dir "$OUT_DIR"
)

if [ "$DRY_RUN" = "1" ]; then
    echo "   ${LV_CMD[*]}"
else
    log "== [1/2] label-invariant experiment =="
    run "${LV_CMD[@]}"
    echo "label-invariant experiment has done!"
fi
echo

# ---------------------------------------------------------------------------
#  2. TabMDA reference (train.py)
# ---------------------------------------------------------------------------
TABMDA_LOG="$LOGS_DIR/tabmda_${DATASET}_n${NUM_REAL_SAMPLES}_r${REPEAT_ID}_nc${NUM_CONTEXTS}_cs${CONTEXT_SIZE}.log"

TABMDA_CMD=(
    "${PY_CMD[@]}"
    train.py
    --dataset "$DATASET"
    --num_real_samples "$NUM_REAL_SAMPLES"
    --repeat_id "$REPEAT_ID"
    --augmentor_model tabmda_encoder
    --classifier_model "$CLASSIFIER_MODEL"
    --num_contexts "$NUM_CONTEXTS"
    --context_size "$CONTEXT_SIZE"
    --train_epochs "$TRAIN_EPOCHS"
)

if [ "$DRY_RUN" = "1" ]; then
    echo "   ${TABMDA_CMD[*]} > $(basename "$TABMDA_LOG")"
else
    log "== [2/2] TabMDA reference =="
    if run "${TABMDA_CMD[@]}" > "$TABMDA_LOG" 2>&1; then
        echo "TabMDA has done!"
    else
        fail "TabMDA run failed (see $TABMDA_LOG)"
    fi
fi
echo

# ---------------------------------------------------------------------------
#  Side-by-side summary (label-invariant JSON vs. TabMDA log)
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

LV_JSON="$OUT_DIR/label_invariant_${DATASET}_r${REPEAT_ID}_sc_${SELF_CONTEXT}.json"
if [ ! -s "$LV_JSON" ]; then
    fail "missing label-invariant output $LV_JSON"
    exit 1
fi

# Extract the TabMDA test balanced accuracy from train.py's log line, e.g.
#   'test/balanced_accuracy': 0.7523,
TABMDA_TEST=$(grep -oE "test/balanced_accuracy': [0-9.eE+-]+" "$TABMDA_LOG" 2>/dev/null | head -1 | grep -oE "[0-9.eE+-]+$")

python - "$LV_JSON" "$TABMDA_TEST" "$NUM_CONTEXTS" "$CONTEXT_SIZE" <<'PY'
import json, sys

lv = json.load(open(sys.argv[1]))
tabmda_test = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
tabmda_nc   = sys.argv[3] if len(sys.argv) > 3 else "?"
tabmda_cs   = sys.argv[4] if len(sys.argv) > 4 else "?"

topk = lv.get("top_k_by_probability", {})
ref  = lv.get("tabmda_reference", {})
per  = lv.get("per_cell", {})

print()
print("=" * 72)
print("Label-invariant Top-K (this scheme)  vs.  TabMDA reference (context_size=0.5)")
print("=" * 72)
hdr = f"{'k':>4}  {'top-k bacc':>12}  {'tabmda_ref bacc':>16}  {'delta':>8}"
print(hdr)
print("-" * 72)
for k in sorted(topk, key=int):
    t = topk[k]["balanced_accuracy"]
    r = ref.get(k, {}).get("balanced_accuracy", float("nan"))
    d = t - r
    print(f"{k:>4}  {t:>12.4f}  {r:>16.4f}  {d:>+8.4f}")

print()
print("per-cell (num_contexts x context_size) downstream bacc:")
ncs = sorted({int(k.split('_')[0][2:]) for k in per})
css = sorted({float(k.split('_')[1][2:]) for k in per})
print(f"{'cs\\nc':>8}" + "".join(f"{nc:>10}" for nc in ncs))
for cs in css:
    row = f"{cs:>8}"
    for nc in ncs:
        key = f"nc{nc}_cs{cs}"
        row += f"{per[key]['balanced_accuracy']:>10.4f}"
    print(row)

if tabmda_test:
    print()
    print(f"External TabMDA (train.py, tabmda_encoder nc={tabmda_nc} cs={tabmda_cs}) "
          f"test balanced accuracy = {tabmda_test}")
    print("Note: the internal tabmda_reference arm uses the experiment's own frozen")
    print("encode + LogReg path; the external train.py value is the real TabMDA pipeline.")
PY

log "done."
log "label-invariant JSON : $LV_JSON"
log "tabmda log           : $TABMDA_LOG"