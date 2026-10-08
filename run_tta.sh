#!/usr/bin/env bash
# ============================================================================
#  TabMDA test-time augmentation (TTA) comparison batch runner
#
#  Compares two test-time-augmentation strategies for the tabmda_encoder arm.
#  Both use TabMDA's in-context subsetting for AUGMENTATION (train encoding), and
#  differ only in what context is used at INFERENCE (val/test encoding):
#
#    strategy                          augmentation context    inference context
#    -------------------------------   ---------------------   -----------------
#    1. tta_subset   (TabMDA-default)  subset of train set     subset of train set
#                                          (grid-searched)       (num_contexts_test)
#    2. tta_full     (full-train)      subset of train set     FULL train set
#                                          + one full-train        (num_contexts=1)
#                                            context retained
#
#  Strategy 1 is exactly the `tabmda_encoder` arm of run_comparison.sh extended
#  so that inference also ensembles over multiple subset contexts
#  (num_contexts_test = NUM_CONTEXTS_TEST, fixed for the whole sweep).
#
#  Strategy 2 adds --ensure_full_train_context: during augmentation, ONE of the
#  num_contexts subset contexts is replaced by a FULL-train-set context, and
#  inference always uses the full train set (num_contexts_test = 1).
#
#  Both strategies are GRID-SEARCHED identically over the in-context subsetting
#  hyperparameters used for augmentation:
#      context_size  in CONTEXT_SIZES     (default 0.5 0.7 0.9 1)
#      num_contexts  in NUM_CONTEXTS_GRID (default 5 20 50)
#  Following run_comparison.sh, the single best (context_size, num_contexts) per
#  (dataset, n, strategy) is selected by mean VALIDATION balanced accuracy, and
#  only that point's TEST score is reported.
#
#  Results are parsed from each run and aggregated over repeats as
#  MEAN +/- STD (sample std, ddof=1) of train/val/test balanced accuracy.
#
#  Usage:
#      bash run_tta.sh              # run everything with defaults
#      DRY_RUN=1 bash run_tta.sh    # print commands only, run nothing
#
#  All settings can be overridden via environment variables, e.g.:
#      DATASETS="vehicle texture" NUM_REAL_SAMPLES="50 100" CONTEXT_SIZES="0.5 0.9" \
#      NUM_CONTEXTS_GRID="5 20" NUM_CONTEXTS_TEST="5" bash run_tta.sh
# ============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-tabmda}"
PYTHON_BIN="${PYTHON_BIN:-}"

# ---------------------------------------------------------------------------
#  Global experiment settings
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LogReg}"

# Context subsetting hyperparameters (grid-applied to BOTH strategies identically).
CONTEXT_SIZES="${CONTEXT_SIZES:-0.5 0.7 0.9 1}"   # context size = proportion of train set
NUM_CONTEXTS_GRID="${NUM_CONTEXTS_GRID:-5 20 50}" # number of subcontexts per sample (augmentation)

# Strategy 1 inference: number of subset contexts ensembled over at test time.
NUM_CONTEXTS_TEST="${NUM_CONTEXTS_TEST:-5}"
AGGREGATION_TEST="${AGGREGATION_TEST:-mean}"

ENABLE_WANDB="${ENABLE_WANDB:-0}"

REPEATS="${REPEATS:-0 1 2 3 4 5 6 7 8 9}"

# ---------------------------------------------------------------------------
#  Datasets and num_real_samples
# ---------------------------------------------------------------------------
DATASETS="${DATASETS:-qsar-biodeg vehicle texture steel-plates-fault MiceProtein mfeat-fourier}"
NUM_REAL_SAMPLES="${NUM_REAL_SAMPLES:-50}"

# Per-dataset sample sizes. Format: space-separated `dataset:n1,n2,...` entries.
# When set, takes precedence over DATASETS / NUM_REAL_SAMPLES.
DATASET_CONFIGS="${DATASET_CONFIGS:-}"

# ---------------------------------------------------------------------------
#  Build the (dataset, num_real_samples) work list
# ---------------------------------------------------------------------------
WORK_ITEMS=""
if [ -n "$DATASET_CONFIGS" ]; then
    for spec in $DATASET_CONFIGS; do
        d="${spec%%:*}"
        ns="${spec#*:}"
        for nrs in $(echo "$ns" | tr ',' ' '); do
            WORK_ITEMS="$WORK_ITEMS $d:$nrs"
        done
    done
else
    for dataset in $DATASETS; do
        for nrs in $NUM_REAL_SAMPLES; do
            WORK_ITEMS="$WORK_ITEMS $dataset:$nrs"
        done
    done
fi

# ---------------------------------------------------------------------------
#  The 2 TTA strategies
# ---------------------------------------------------------------------------
ARMS="${ARMS:-tta_subset tta_full}"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
#  Output
# ---------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-results_tta}"
LOGS_DIR="$OUTPUT_DIR/logs"
RUN_DATASETS=""

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
log()  { echo "[run_tta] $*"; }
fail() { echo "[run_tta] ERROR: $*" >&2; }

resolve_python() {
    if [ -n "$PYTHON_BIN" ]; then
        PY_CMD=("$PYTHON_BIN")
    elif [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
        PY_CMD=(python)
    elif command -v conda >/dev/null 2>&1; then
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

extract_metric() {
    grep -oE "$2': [0-9.eE+-]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9.eE+-]+$"
}

# Run a single experiment and record its metrics into $OUTPUT_DIR/<dataset>.tsv.
run_one() {
    local dataset="$1" nrs="$2" repeat="$3" arm="$4" cs="$5" nc="$6"

    log "== [$TOTAL] dataset=$dataset  n=$nrs  repeat=$repeat  strategy=$arm  cs=$cs  nc=$nc =="

    local -a extra=(--augmentor_model tabmda_encoder
                    --context_size "$cs" --num_contexts "$nc"
                    --aggregation_over_contexts_test "$AGGREGATION_TEST")
    case "$arm" in
        tta_subset)
            extra+=(--num_contexts_val 1 --num_contexts_test "$NUM_CONTEXTS_TEST")
            ;;
        tta_full)
            extra+=(--num_contexts_val 1 --num_contexts_test 1
                    --ensure_full_train_context)
            ;;
        *)
            fail "unknown strategy: $arm"
            return 1
            ;;
    esac

    if [ "$DRY_RUN" = "1" ]; then
        echo "   ${PY_CMD[*]} train.py --dataset \"$dataset\" --num_real_samples $nrs \
--repeat_id $repeat --classifier_model $CLASSIFIER_MODEL ${extra[*]}"
        return 0
    fi

    local log_file="$LOGS_DIR/${dataset}_n${nrs}_r${repeat}_${arm}_cs${cs}_nc${nc}.log"

    run "${PY_CMD[@]}" train.py \
        --dataset "$dataset" \
        --num_real_samples "$nrs" \
        --repeat_id "$repeat" \
        --classifier_model "$CLASSIFIER_MODEL" \
        "${extra[@]}" > "$log_file" 2>&1

    local rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "run failed (exit $rc): dataset=$dataset n=$nrs repeat=$repeat strategy=$arm cs=$cs nc=$nc (see $log_file)"
        FAILED=$((FAILED + 1))
        echo
        return 1
    fi

    local train_acc val_acc test_acc
    train_acc=$(extract_metric "$log_file" "train/balanced_accuracy")
    val_acc=$(extract_metric "$log_file" "val/balanced_accuracy")
    test_acc=$(extract_metric "$log_file" "test/balanced_accuracy")

    if [ -n "$test_acc" ]; then
        local out_file="$OUTPUT_DIR/${dataset}.tsv"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dataset" "$nrs" "$arm" "$cs" "$nc" "$repeat" "$train_acc" "$val_acc" "$test_acc" >> "$out_file"
    else
        fail "no metrics parsed from $log_file"
        FAILED=$((FAILED + 1))
    fi
    echo
}

# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
resolve_python
log "python : ${PY_CMD[*]}"
log "classifier: $CLASSIFIER_MODEL  |  wandb: $ENABLE_WANDB  |  dry-run: $DRY_RUN"
if [ -n "$DATASET_CONFIGS" ]; then
    log "datasets : (per-dataset) $DATASET_CONFIGS"
else
    log "datasets : $DATASETS"
    log "n_real   : $NUM_REAL_SAMPLES"
fi
log "work items: $WORK_ITEMS"
log "repeats  : $REPEATS"
log "strategies: $ARMS"
log "grid     : context_size=[$CONTEXT_SIZES]  x  num_contexts=[$NUM_CONTEXTS_GRID]"
log "inference: tta_subset ensembles over num_contexts_test=$NUM_CONTEXTS_TEST | tta_full uses full train set"
echo

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"
    for item in $WORK_ITEMS; do
        d="${item%%:*}"
        case " $RUN_DATASETS " in
            *" $d "*) ;;
            *) RUN_DATASETS="$RUN_DATASETS $d" ;;
        esac
    done
    for d in $RUN_DATASETS; do
        printf 'dataset\tn\tarm\tcontext_size\tnum_contexts\trepeat\ttrain\tval\ttest\n' > "$OUTPUT_DIR/${d}.tsv"
    done
    log "writing per-run logs to $LOGS_DIR"
    log "writing parsed metrics to $OUTPUT_DIR/<dataset>.tsv"
    echo
fi

FAILED=0
TOTAL=0

for item in $WORK_ITEMS; do
    dataset="${item%%:*}"
    nrs="${item#*:}"
    for repeat in $REPEATS; do
        for arm in $ARMS; do
            for cs in $CONTEXT_SIZES; do
                for nc in $NUM_CONTEXTS_GRID; do
                    TOTAL=$((TOTAL + 1))
                    run_one "$dataset" "$nrs" "$repeat" "$arm" "$cs" "$nc"
                done
            done
        done
    done
done

log "done. total=$TOTAL  failed=$FAILED"
log "raw results: $OUTPUT_DIR/<dataset>.tsv"
log "per-run logs: $LOGS_DIR"
echo

# ---------------------------------------------------------------------------
#  Aggregate over repeats -> mean +/- std
#  (per strategy: first select best grid point by mean val, then report it)
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" != "1" ]; then
    for d in $RUN_DATASETS; do
        local_file="$OUTPUT_DIR/${d}.tsv"
        if [ ! -s "$local_file" ]; then
            continue
        fi
        log "aggregating over repeats (mean +/- sample std) for $d..."
        python - "$local_file" <<'PY'
import sys, math

rows = {}
with open(sys.argv[1]) as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 9:
            continue
        dataset, n, arm, cs, nc, repeat, train, val, test = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
        try:
            train, val, test = float(train), float(val), float(test)
        except ValueError:
            continue
        rows.setdefault((dataset, int(n), arm, cs, nc), []).append((train, val, test))

def meanstd(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return m, s

# Select best (context_size, num_contexts) per (dataset, n, arm) by mean val accuracy.
chosen = {}   # (dataset, n, arm) -> (val_mean, cs, nc)
for (dataset, n, arm, cs, nc), vals in rows.items():
    vm = sum(v[1] for v in vals) / len(vals)
    key = (dataset, n, arm)
    if key not in chosen or vm > chosen[key][0]:
        chosen[key] = (vm, cs, nc)

order_arms = ["tta_subset", "tta_full"]
arm_rank = {a: i for i, a in enumerate(order_arms)}
display_names = {"tta_subset": "tta_subset (subset ctx)", "tta_full": "tta_full (full-train ctx)"}

display = []
for (dataset, n, arm, cs, nc), vals in rows.items():
    if chosen.get((dataset, n, arm), (None, None, None))[1:] != (cs, nc):
        continue  # only report the selected grid point
    display.append((dataset, n, arm, cs, nc, vals))

display.sort(key=lambda k: (k[0], k[1], arm_rank.get(k[2], 99)))

print()
print(f"{'dataset':<22}{'n':>5}  {'strategy':<26}{'rep':>4}  {'train':>18}  {'val':>18}  {'test':>18}")
print("-" * 110)
for dataset, n, arm, cs, nc, vals in display:
    cnt = len(vals)
    tm, ts = meanstd([v[0] for v in vals])
    vm, vs = meanstd([v[1] for v in vals])
    em, es = meanstd([v[2] for v in vals])
    label = f"{display_names.get(arm, arm)} (cs={cs}, nc={nc})"
    print(f"{dataset:<22}{n:>5}  {label:<26}{cnt:>4}  {tm:>9.4f}+-{ts:<7.4f}  {vm:>9.4f}+-{vs:<7.4f}  {em:>9.4f}+-{es:<7.4f}")
print()

# Summary of chosen settings
if chosen:
    print("Selected (context_size, num_contexts) per (dataset, n, strategy) by best mean val balanced accuracy:")
    for (dataset, n, arm), (vm, cs, nc) in sorted(chosen.items()):
        print(f"  {dataset:<22} n={n:>4}  {arm:<12} ->  context_size={cs}  num_contexts={nc}  (mean val = {vm:.4f})")
print()
print("rep = number of successful repeats aggregated. std is sample std (ddof=1).")
print("Both strategies are grid-searched; the best setting is chosen by mean val balanced accuracy.")
PY
        echo
    done
fi

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi