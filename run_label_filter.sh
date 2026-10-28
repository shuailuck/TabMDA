#!/usr/bin/env bash
# ============================================================================
#  Label-filter experiment: original TabMDA vs. label-invariant TabMDA
#  (GRID-SEARCHED, TabMDA-style)
#
#  Mirroring run_comparison.sh's protocol for the tabmda_encoder arm, but for
#  the label-invariance filter comparison. For each (dataset, n, repeat) we try
#  every combination of
#      context_size  in CONTEXT_SIZES      (default 0.5 0.7 0.9)
#      num_contexts  in NUM_CONTEXTS_GRID   (default 5 20 50)
#  and, once all repeats are collected, select the setting with the best mean
#  VALIDATION balanced accuracy. Only that setting's TEST score is reported.
#
#  Two arms, each grid-searched independently (so the comparison is fair —
#  every arm gets its own best (cs, nc)):
#
#    * tabmda      - original TabMDA (train.py --augmentor_model tabmda_encoder).
#    * filtered    - per-query leak-free context generator that keeps sampling
#                    contexts until exactly K satisfy
#                    argmax P(y | TabPFN(x_i, C)) == y_i (label-invariant).
#                    Queries that can't reach K valid contexts within
#                    `max_attempts` are dropped (reported).
#
#  Both arms use the SAME frozen-TabPFN embedding + LogisticRegression
#  downstream; the val/test protocols are likewise matched. The
#  train.py `tabmda` arm is the external "original TabMDA" anchor.
#
#  Usage:
#      bash run_label_filter.sh                        # vehicle@100, grid 0.5/0.7/0.9 x 5/20/50
#      DRY_RUN=1 bash run_label_filter.sh              # print commands only
#      DATASETS="vehicle texture" REPEATS="0 1 2" bash run_label_filter.sh
#      CONTEXT_SIZES="0.5 0.9" NUM_CONTEXTS_GRID="5 20" bash run_label_filter.sh
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
#  Datasets and num_real_samples
# ---------------------------------------------------------------------------
DATASETS="${DATASETS:-vehicle}"
NUM_REAL_SAMPLES="${NUM_REAL_SAMPLES:-100}"
REPEATS="${REPEATS:-0}"

# ---------------------------------------------------------------------------
#  Grid (context_size x num_contexts), selected by mean val over repeats
# ---------------------------------------------------------------------------
CONTEXT_SIZES="${CONTEXT_SIZES:-0.5 0.7 0.9}"
NUM_CONTEXTS_GRID="${NUM_CONTEXTS_GRID:-5 20 50}"

# ---------------------------------------------------------------------------
#  Label-filter specifics (experiment_label_filter.py fixed-cell mode)
# ---------------------------------------------------------------------------
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"        # hard cap on sampling attempts/query while filtering
SELF_CONTEXT="${SELF_CONTEXT:-exclude}"    # leak-free per-query contexts (required for the filter)
SEED="${SEED:-42}"

# ---------------------------------------------------------------------------
#  TabMDA reference (train.py)
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LogReg}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
NUM_CONTEXTS_VAL="${NUM_CONTEXTS_VAL:-1}"
NUM_CONTEXTS_TEST="${NUM_CONTEXTS_TEST:-1}"
AGGREGATION_TEST="${AGGREGATION_TEST:-mean}"

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

# extract_kv: pull a key's value from the RESULT_FILTERED line, e.g.
#   'RESULT_FILTERED val=0.7523 test=0.71 kept=20 dropped=5'
# Anchored to the RESULT_FILTERED line so the [setup] line's val/test sizes
# are not mistaken for the downstream accuracies.
extract_kv() {
    grep -oE "RESULT_FILTERED .*" "$1" 2>/dev/null | head -1 \
        | grep -oE "$2=[0-9.eE+-]+" | head -1 | grep -oE "[0-9.eE+-]+$"
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
log "python      : ${PY_CMD[*]}"
log "datasets    : $DATASETS   n=$NUM_REAL_SAMPLES   repeats=$REPEATS"
log "grid        : context_size=[$CONTEXT_SIZES]  x  num_contexts=[$NUM_CONTEXTS_GRID]"
log "filter      : max_attempts=$MAX_ATTEMPTS  self_context=$SELF_CONTEXT  seed=$SEED"
log "tabmda      : classifier=$CLASSIFIER_MODEL  epochs=$TRAIN_EPOCHS  wandb=$ENABLE_WANDB"
if [ "$DRY_RUN" = "1" ]; then
    log "dry-run: commands below are NOT executed."
fi
echo

mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"

# Fresh per-dataset TSV holding the tabmda + filtered arm rows.
# Columns: dataset  n  repeat  cs  nc  arm  val  test
for d in $DATASETS; do
    printf 'dataset\tn\trepeat\tcs\tnc\tarm\tval\ttest\n' > "$OUTPUT_DIR/${d}.tsv"
done

FAILED=0

for item in $WORK_ITEMS; do
    dataset="${item%%:*}"
    n="${item#*:}"
    for repeat in $REPEATS; do
        for cs in $CONTEXT_SIZES; do
            for nc in $NUM_CONTEXTS_GRID; do

                # ===== Arm A: original TabMDA (train.py, grid cell) =====
                tabmda_log="$LOGS_DIR/tabmda_${dataset}_n${n}_r${repeat}_cs${cs}_nc${nc}.log"
                if [ "$DRY_RUN" = "1" ]; then
                    echo "   ${PY_CMD[*]} train.py --dataset $dataset --num_real_samples $n --repeat_id $repeat \
--augmentor_model tabmda_encoder --classifier_model $CLASSIFIER_MODEL \
--context_size $cs --num_contexts $nc --train_epochs $TRAIN_EPOCHS > $tabmda_log"
                else
                    log "== [$dataset n=$n r=$repeat cs=$cs nc=$nc] original TabMDA =="
                    if run "${PY_CMD[@]}" train.py \
                        --dataset "$dataset" --num_real_samples "$n" --repeat_id "$repeat" \
                        --augmentor_model tabmda_encoder --classifier_model "$CLASSIFIER_MODEL" \
                        --context_size "$cs" --num_contexts "$nc" \
                        --num_contexts_val "$NUM_CONTEXTS_VAL" --num_contexts_test "$NUM_CONTEXTS_TEST" \
                        --aggregation_over_contexts_test "$AGGREGATION_TEST" \
                        --train_epochs "$TRAIN_EPOCHS" \
                        > "$tabmda_log" 2>&1; then
                        tv=$(extract_metric "$tabmda_log" "val/balanced_accuracy")
                        te=$(extract_metric "$tabmda_log" "test/balanced_accuracy")
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                            "$dataset" "$n" "$repeat" "$cs" "$nc" "tabmda" "${tv:-nan}" "${te:-nan}" \
                            >> "$OUTPUT_DIR/${dataset}.tsv"
                    else
                        fail "train.py failed: $dataset n=$n r=$repeat cs=$cs nc=$nc (see $tabmda_log)"
                        FAILED=$((FAILED + 1))
                    fi
                fi

                # ===== Arm B: label-filter (filtered only, at this cell) =====
                filter_log="$LOGS_DIR/filter_${dataset}_n${n}_r${repeat}_cs${cs}_nc${nc}.log"
                lv_cmd=("${PY_CMD[@]}" experiment_label_filter.py \
                    --dataset "$dataset" --num_real_samples "$n" --repeat_id "$repeat" \
                    --context_size "$cs" --num_contexts "$nc" --max_attempts "$MAX_ATTEMPTS" \
                    --self_context "$SELF_CONTEXT" --seed "$SEED" --no_json \
                    --out_dir "$OUTPUT_DIR" --cache_dir "$OUTPUT_DIR/cache")
                if [ "$DRY_RUN" = "1" ]; then
                    echo "   ${lv_cmd[*]}"
                else
                    log "== [$dataset n=$n r=$repeat cs=$cs nc=$nc] label filter =="
                    if run "${lv_cmd[@]}" > "$filter_log" 2>&1; then
                        fv=$(extract_kv "$filter_log" "val")
                        fe=$(extract_kv "$filter_log" "test")
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                            "$dataset" "$n" "$repeat" "$cs" "$nc" "filtered" "${fv:-nan}" "${fe:-nan}" \
                            >> "$OUTPUT_DIR/${dataset}.tsv"
                    else
                        fail "experiment_label_filter.py failed: $dataset n=$n r=$repeat cs=$cs nc=$nc (see $filter_log)"
                        FAILED=$((FAILED + 1))
                    fi
                fi
                echo
            done
        done
    done
done

[ "$DRY_RUN" = "1" ] && exit 0

# ---------------------------------------------------------------------------
#  Aggregate over repeats -> mean +/- std, per arm; select best cell by val.
#
#  Reads both arms (tabmda + filtered) from $OUTPUT_DIR/<dataset>.tsv, then:
#    (1) groups by (n, arm, cs, nc) over repeats,
#    (2) selects the (cs, nc) with the best mean VAL balanced accuracy,
#    (3) reports that cell's TEST balanced accuracy as mean +/- std.
# ---------------------------------------------------------------------------
for item in $WORK_ITEMS; do
    dataset="${item%%:*}"
    n="${item#*:}"
    tsv="$OUTPUT_DIR/${dataset}.tsv"
    [ -s "$tsv" ] || continue
    log "aggregating $dataset ..."
    python - "$tsv" "$OUTPUT_DIR" "$dataset" "$n" <<'PY'
import sys, math

tsv_path, out_dir, dataset, n_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
n = int(n_str)

# rows[(arm, cs, nc)] = list of (val, test)
rows = {}
with open(tsv_path) as f:
    header = f.readline().rstrip('\n').split('\t')
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) < 8:
            continue
        d, rn, rep, cs, nc, arm, val, test = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
        if d != dataset or int(rn) != n:
            continue
        try:
            val, test = float(val), float(test)
        except ValueError:
            continue
        rows.setdefault((arm, cs, nc), []).append((val, test))

def meanstd(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return m, s

# --- select best (cs, nc) per arm by mean val ---
best = {}  # arm -> (val_mean, cs, nc, list_of_(val,test))
for (arm, cs, nc), vals in rows.items():
    vm = sum(v[0] for v in vals) / len(vals)
    if arm not in best or vm > best[arm][0]:
        best[arm] = (vm, cs, nc, vals)

order = ["tabmda", "filtered"]

print()
print("=" * 96)
print("Label-invariant filter vs. original TabMDA  (grid-searched, best cell by val)")
print("=" * 96)
print(f"  dataset={dataset}  n={n}")
print("-" * 96)
print(f"{'arm':<14}{'best cs':>10}{'best nc':>10}{'rep':>5}  {'val bacc':>18}  {'test bacc':>18}")
print("-" * 96)
for arm in order:
    if arm not in best:
        print(f"{arm:<14}{'-':>10}{'-':>10}{0:>5}  {'nan':>18}  {'nan':>18}")
        continue
    vm, cs, nc, vals = best[arm]
    cnt = len(vals)
    vmu, vsd = meanstd([v[0] for v in vals])
    tmu, tsd = meanstd([v[1] for v in vals])
    print(f"{arm:<14}{cs:>10}{nc:>10}{cnt:>5}  {vmu:>9.4f}+-{vsd:<7.4f}  {tmu:>9.4f}+-{tsd:<7.4f}")
print("-" * 96)

print()
print("Selected settings:")
for arm in order:
    if arm in best:
        print(f"  {arm:<14} cs={best[arm][1]} nc={best[arm][2]}")
print()
print("rep = number of successful repeats aggregated. std is sample std (ddof=1).")
print("Each arm is grid-searched independently; the best (context_size, num_contexts)")
print("is chosen by mean VAL balanced accuracy over repeats.")
PY
    echo
done

if [ "$FAILED" -ne 0 ]; then
    log "done with $FAILED failure(s)."
    exit 1
fi
log "done. results: $OUTPUT_DIR/<dataset>.tsv   |   JSON: $OUTPUT_DIR/label_filter_*.json   |   logs: $LOGS_DIR"