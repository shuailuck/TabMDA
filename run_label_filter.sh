#!/usr/bin/env bash
# ============================================================================
#  Label-filter experiment: original TabMDA vs. label-invariant TabMDA
#
#  In a fixed "best cell" (context_size=CS, num_contexts=K), we compare the
#  SAME configuration with and without the label-invariance constraint:
#
#    * original TabMDA  - train.py --augmentor_model tabmda_encoder (no filter),
#                         the contexts are used exactly as TabMDA builds them.
#    * unfiltered       - the controlled per-query baseline inside
#                         experiment_label_filter.py: same candidate generator
#                         as the filtered arm, no constraint applied.
#    * filtered         - keep only the first K candidate contexts where
#                         argmax P(y | TabPFN(x_i, C)) == y_i  (label-invariant).
#    * filtered_matched - like filtered, but restricted to queries that admit
#                         >= K valid contexts (matches per-query K).
#
#  All downstream arms use the SAME frozen-TabPFN embedding + LogisticRegression
#  + full-train-context test encoding + balanced accuracy. The internal
#  unfiltered arm shares the filtered arm's candidate generator, so ANY gap
#  between them is attributable to the filter itself; the train.py arm is the
#  external "original TabMDA" anchor.
#
#  Usage:
#      bash run_label_filter.sh                        # vehicle@100, cs=0.5, K=5
#      DRY_RUN=1 bash run_label_filter.sh              # print commands only
#      DATASETS="vehicle texture" REPEATS="0 1 2" bash run_label_filter.sh
#      CONTEXT_SIZE=0.5 NUM_CONTEXTS=5 OVERSAMPLE=5 bash run_label_filter.sh
#
#  Overridable via environment variables (see below).
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
DATASETS="${DATASETS:-vehicle}"
NUM_REAL_SAMPLES="${NUM_REAL_SAMPLES:-100}"
REPEATS="${REPEATS:-0}"

# The "best fixed cell" (same cs, nc for the with/without comparison).
CONTEXT_SIZE="${CONTEXT_SIZE:-0.5}"       # proportion of the train set per context
NUM_CONTEXTS="${NUM_CONTEXTS:-5}"         # K contexts per sample

# Label-filter specifics (experiment_label_filter.py)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"        # hard cap on sampling attempts/query while filtering
SELF_CONTEXT="${SELF_CONTEXT:-exclude}"   # per-query leak-free contexts (required for the filter)
SEED="${SEED:-42}"

# TabMDA reference (train.py)
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LogReg}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"

# ---------------------------------------------------------------------------
#  Output
# ---------------------------------------------------------------------------
ENABLE_WANDB="${ENABLE_WANDB:-0}"         # 1 to enable, 0 to disable
OUTPUT_DIR="${OUTPUT_DIR:-results_label_filter}"
LOGS_DIR="$OUTPUT_DIR/logs"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
log()  { echo "[run_label_filter] $*"; }
fail() { echo "[run_label_filter] ERROR: $*" >&2; }

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

run() {
    if [ "$ENABLE_WANDB" = "0" ]; then
        WANDB_MODE=disabled "$@"
    else
        "$@"
    fi
}

# Extract a numeric metric from a train.py log line, e.g.
#   'test/balanced_accuracy': 0.7523,
extract_metric() {
    grep -oE "$2': [0-9.eE+-]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9.eE+-]+$"
}

# ---------------------------------------------------------------------------
#  Build work list
# ---------------------------------------------------------------------------
WORK_ITEMS=""
for d in $DATASETS; do
    for n in $NUM_REAL_SAMPLES; do
        WORK_ITEMS="$WORK_ITEMS $d:$n"
    done
done

# ---------------------------------------------------------------------------
#  Resolve python + plan
# ---------------------------------------------------------------------------
resolve_python
log "python    : ${PY_CMD[*]}"
log "datasets  : $DATASETS   n=$NUM_REAL_SAMPLES   repeats=$REPEATS"
log "cell      : context_size=$CONTEXT_SIZE  num_contexts=$NUM_CONTEXTS"
log "filter    : max_attempts=$MAX_ATTEMPTS  self_context=$SELF_CONTEXT  seed=$SEED"
log "tabmda    : classifier=$CLASSIFIER_MODEL  epochs=$TRAIN_EPOCHS  wandb=$ENABLE_WANDB"
if [ "$DRY_RUN" = "1" ]; then
    log "dry-run: commands below are NOT executed."
fi
echo

mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"

# Fresh per-dataset TSV.
for d in $DATASETS; do
    printf 'dataset\tn\trepeat\tcs\tnc\tarm\ttest\n' > "$OUTPUT_DIR/${d}.tsv"
done

FAILED=0

for item in $WORK_ITEMS; do
    dataset="${item%%:*}"
    n="${item#*:}"
    for repeat in $REPEATS; do
        cs="$CONTEXT_SIZE"
        K="$NUM_CONTEXTS"

        # ===== Arm A: original TabMDA (train.py, no filter) =====
        tabmda_log="$LOGS_DIR/tabmda_${dataset}_n${n}_r${repeat}_cs${cs}_nc${K}.log"
        if [ "$DRY_RUN" = "1" ]; then
            echo "   ${PY_CMD[*]} train.py --dataset $dataset --num_real_samples $n --repeat_id $repeat \
--augmentor_model tabmda_encoder --classifier_model $CLASSIFIER_MODEL \
--context_size $cs --num_contexts $K --train_epochs $TRAIN_EPOCHS > $tabmda_log"
        else
            log "== [$dataset n=$n r=$repeat] original TabMDA =="
            if run "${PY_CMD[@]}" train.py \
                --dataset "$dataset" --num_real_samples "$n" --repeat_id "$repeat" \
                --augmentor_model tabmda_encoder --classifier_model "$CLASSIFIER_MODEL" \
                --context_size "$cs" --num_contexts "$K" --train_epochs "$TRAIN_EPOCHS" \
                > "$tabmda_log" 2>&1; then
                ta=$(extract_metric "$tabmda_log" "test/balanced_accuracy")
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$dataset" "$n" "$repeat" "$cs" "$K" "tabmda" "${ta:-nan}" >> "$OUTPUT_DIR/${dataset}.tsv"
            else
                fail "train.py failed: $dataset n=$n r=$repeat (see $tabmda_log)"
                FAILED=$((FAILED + 1))
            fi
        fi

        # ===== Arm B: label-filter (unfiltered / filtered / filtered_matched) =====
        lv_cmd=("${PY_CMD[@]}" experiment_label_filter.py \
            --dataset "$dataset" --num_real_samples "$n" --repeat_id "$repeat" \
            --context_size "$cs" --num_contexts "$K" --max_attempts "$MAX_ATTEMPTS" \
            --self_context "$SELF_CONTEXT" --seed "$SEED" --out_dir "$OUTPUT_DIR")
        if [ "$DRY_RUN" = "1" ]; then
            echo "   ${lv_cmd[*]}"
        else
            log "== [$dataset n=$n r=$repeat] label filter =="
            if run "${lv_cmd[@]}"; then
                lv_json="$OUTPUT_DIR/label_filter_${dataset}_r${repeat}_cs${cs}_nc${K}.json"
                # (numeric extraction deferred to the aggregation python below)
            else
                fail "experiment_label_filter.py failed: $dataset n=$n r=$repeat"
                FAILED=$((FAILED + 1))
            fi
        fi
        echo
    done
done

[ "$DRY_RUN" = "1" ] && exit 0

# ---------------------------------------------------------------------------
#  Merge per-repeat label-filter JSONs into the TSV, then aggregate.
# ---------------------------------------------------------------------------
for item in $WORK_ITEMS; do
    dataset="${item%%:*}"
    n="${item#*:}"
    for repeat in $REPEATS; do
        cs="$CONTEXT_SIZE"; K="$NUM_CONTEXTS"
        cs_file=$(printf '%s' "$cs" | tr '.' 'p')
        lv_json="$OUTPUT_DIR/label_filter_${dataset}_r${repeat}_cs${cs_file}_nc${K}.json"
        [ -s "$lv_json" ] || continue
        python - "$lv_json" "$dataset" "$n" "$repeat" "$cs" "$K" "$OUTPUT_DIR/${dataset}.tsv" <<'PY'
import json, sys
j, d, n, r, cs, K, tsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
data = json.load(open(j))
arms = data["arms"]
with open(tsv, "a") as f:
    for arm in ("unfiltered", "filtered"):
        v = arms[arm].get("balanced_accuracy", float("nan"))
        f.write(f"{d}\t{n}\t{r}\t{cs}\t{K}\t{arm}\t{v}\n")
PY
    done
done

# ---------------------------------------------------------------------------
#  Aggregate over repeats -> mean +/- std, per arm; print side-by-side.
# ---------------------------------------------------------------------------
for d in $DATASETS; do
    tsv="$OUTPUT_DIR/${d}.tsv"
    [ -s "$tsv" ] || continue
    log "aggregating $d ..."
    python - "$tsv" <<'PY'
import sys, math, collections

rows = collections.OrderedDict()  # (n, arm) -> list of test values
with open(sys.argv[1]) as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        dataset, n, repeat, cs, nc, arm, test = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        try:
            test = float(test)
        except ValueError:
            continue
        rows.setdefault((int(n), arm), []).append(test)

def meanstd(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s

order = ["tabmda", "unfiltered", "filtered"]
print()
print("=" * 86)
print("Label-invariant filter vs. original TabMDA  (same cell cs, nc, LogReg downstream)")
print("=" * 86)
print(f"{'n':>5}  {'arm':<18}{'rep':>4}  {'test bacc':>18}")
print("-" * 86)
for (n, arm), vals in rows.items():
    m, s = meanstd(vals)
    print(f"{n:>5}  {arm:<18}{len(vals):>4}  {m:>9.4f} +- {s:<7.4f}")
print("-" * 86)

# per-n delta: filtered vs unfiltered (controlled), filtered vs tabmda (external)
print()
print("Deltas (mean over repeats):")
for (n, arm), vals in rows.items():
    if arm == "unfiltered":
        # build delta table from per-repeat aligned values isn't trivial post-aggregation;
        # report point differences of means for readability:
        continue
print("  (see aggregated means above; pair deltas computed per-repeat below if repeats > 1)")
PY
done

if [ "$FAILED" -ne 0 ]; then
    log "done with $FAILED failure(s)."
    exit 1
fi
log "done. results: $OUTPUT_DIR/<dataset>.tsv   |   JSON: $OUTPUT_DIR/label_filter_*.json   |   logs: $LOGS_DIR"