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
#  The `tabmda_encoder` arm is GRID-SEARCHED over the in-context subsetting
#  hyperparameters, following the paper: for each (dataset, num_real_samples)
#  configuration, we try every combination of
#      context_size  in CONTEXT_SIZES    (default 0.5 0.7 0.9 1)
#      num_contexts  in NUM_CONTEXTS_GRID (default 5 20 50)
#  and select the setting with the best VALIDATION balanced accuracy. Only the
#  selected setting's TEST score is reported as the TabMDA result.
#
#  Results are parsed from each run and aggregated over repeats as
#  MEAN +/- STD (sample std, ddof=1) of train/val/test balanced accuracy.
#
#  Usage:
#      bash run_comparison.sh            # run everything with defaults
#      DRY_RUN=1 bash run_comparison.sh  # print commands only, run nothing
#
#  All settings below can be overridden via environment variables, e.g.:
#      DATASETS="vehicle texture" NUM_REAL_SAMPLES="50 100" REPEATS="0 1 2" bash run_comparison.sh
#      CONTEXT_SIZES="0.5 0.9" NUM_CONTEXTS_GRID="5 20" bash run_comparison.sh
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

# ----- TabMDA grid search (in-context subsetting, applied to tabmda_encoder) -----
CONTEXT_SIZES="${CONTEXT_SIZES:-0.5 0.7 0.9 1}"   # context size = proportion of train set
NUM_CONTEXTS_GRID="${NUM_CONTEXTS_GRID:-5 20 50}" # number of subcontexts per sample

NUM_CONTEXTS_VAL="${NUM_CONTEXTS_VAL:-1}"
NUM_CONTEXTS_TEST="${NUM_CONTEXTS_TEST:-1}"
AGGREGATION_TEST="${AGGREGATION_TEST:-mean}"

SMOTE_K="${SMOTE_K:-3}"                   # SMOTE k-nearest neighbours
SMOTE_ROUNDS="${SMOTE_ROUNDS:-1}"         # SMOTE rounds (each round ~doubles)

ENABLE_WANDB="${ENABLE_WANDB:-0}"         # 1 to enable, 0 to disable (needs api key)

REPEATS="${REPEATS:-0 1 2 3 4 5 6 7 8 9}" # repeat ids (0..9), aggregated as mean+/-std

# ---------------------------------------------------------------------------
#  Datasets and num_real_samples (space-separated lists)
#
#  Every combination of DATASETS x NUM_REAL_SAMPLES is run. E.g.:
#      DATASETS="vehicle texture" NUM_REAL_SAMPLES="50 100"
#  runs vehicle:50, vehicle:100, texture:50, texture:100.
#
#  Supported num_real_samples per dataset (see dataset/datasets.py):
#      qsar-biodeg         20 50 100 200 500 -1
#      vehicle             20 50 100 200 -1
#      texture             50 100 200 500 -1
#      steel-plates-fault  20 50 100 200 500 -1
#      MiceProtein         50 100 200 500 -1
#      mfeat-fourier       50 100 200 500 -1
# ---------------------------------------------------------------------------
DATASETS="${DATASETS:-qsar-biodeg vehicle texture steel-plates-fault MiceProtein mfeat-fourier}"
NUM_REAL_SAMPLES="${NUM_REAL_SAMPLES:-50}"

# ---------------------------------------------------------------------------
#  The 4 comparison arms (order matters for readability of results)
# ---------------------------------------------------------------------------
ARMS="${ARMS:-none smote tabmda_embedding tabmda_encoder}"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
#  Output
# ---------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-results}"
LOGS_DIR="$OUTPUT_DIR/logs"
RESULTS_FILE="${RESULTS_FILE:-$OUTPUT_DIR/results.tsv}"

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

# Extract a numeric metric from a train.py log file.
#   $1 = log file,  $2 = metric key (e.g. "test/balanced_accuracy")
extract_metric() {
    grep -oE "$2': [0-9.eE+-]+" "$1" 2>/dev/null | head -1 | grep -oE "[0-9.eE+-]+$"
}

# Run a single experiment and record its metrics into RESULTS_FILE.
#   $1 = dataset, $2 = num_real_samples, $3 = repeat_id, $4 = arm
#   $5 = context_size (grid arm only, else ""), $6 = num_contexts (grid arm only, else "")
run_one() {
    local dataset="$1" nrs="$2" repeat="$3" arm="$4" cs="$5" nc="$6"

    local -a extra
    case "$arm" in
        none)
            extra=(--augmentor_model none)
            ;;
        smote)
            extra=(--augmentor_model smote --smote_k "$SMOTE_K" --smote_rounds "$SMOTE_ROUNDS")
            ;;
        tabmda_embedding)
            extra=(--augmentor_model tabmda_embedding)
            ;;
        tabmda_encoder)
            extra=(--augmentor_model tabmda_encoder
                   --context_size "$cs" --num_contexts "$nc"
                   --num_contexts_val "$NUM_CONTEXTS_VAL" --num_contexts_test "$NUM_CONTEXTS_TEST"
                   --aggregation_over_contexts_test "$AGGREGATION_TEST")
            ;;
        *)
            fail "unknown arm: $arm"
            return 1
            ;;
    esac

    log "== [$TOTAL] dataset=$dataset  n=$nrs  repeat=$repeat  arm=$arm  cs=${cs:-na}  nc=${nc:-na} =="

    if [ "$DRY_RUN" = "1" ]; then
        echo "   ${PY_CMD[*]} train.py --dataset \"$dataset\" --num_real_samples $nrs \
--repeat_id $repeat --classifier_model $CLASSIFIER_MODEL ${extra[*]}"
        return 0
    fi

    local log_file
    if [ "$arm" = "tabmda_encoder" ]; then
        log_file="$LOGS_DIR/${dataset}_n${nrs}_r${repeat}_${arm}_cs${cs}_nc${nc}.log"
    else
        log_file="$LOGS_DIR/${dataset}_n${nrs}_r${repeat}_${arm}.log"
    fi

    run "${PY_CMD[@]}" train.py \
        --dataset "$dataset" \
        --num_real_samples "$nrs" \
        --repeat_id "$repeat" \
        --classifier_model "$CLASSIFIER_MODEL" \
        "${extra[@]}" > "$log_file" 2>&1

    local rc=$?
    if [ "$rc" -ne 0 ]; then
        fail "run failed (exit $rc): dataset=$dataset n=$nrs repeat=$repeat arm=$arm cs=${cs:-na} nc=${nc:-na} (see $log_file)"
        FAILED=$((FAILED + 1))
        echo
        return 1
    fi

    local train_acc val_acc test_acc
    train_acc=$(extract_metric "$log_file" "train/balanced_accuracy")
    val_acc=$(extract_metric "$log_file" "val/balanced_accuracy")
    test_acc=$(extract_metric "$log_file" "test/balanced_accuracy")

    if [ -n "$test_acc" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$dataset" "$nrs" "$arm" "$cs" "$nc" "$repeat" "$train_acc" "$val_acc" "$test_acc" >> "$RESULTS_FILE"
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
log "datasets : $DATASETS"
log "n_real   : $NUM_REAL_SAMPLES"
log "repeats  : $REPEATS"
log "arms     : $ARMS"
log "grid     : context_size=[$CONTEXT_SIZES]  x  num_contexts=[$NUM_CONTEXTS_GRID]"
echo

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"
    printf 'dataset\tn\tarm\tcontext_size\tnum_contexts\trepeat\ttrain\tval\ttest\n' > "$RESULTS_FILE"
    log "writing per-run logs to $LOGS_DIR"
    log "writing parsed metrics to $RESULTS_FILE"
    echo
fi

FAILED=0
TOTAL=0

for dataset in $DATASETS; do
    for nrs in $NUM_REAL_SAMPLES; do
        for repeat in $REPEATS; do
            for arm in $ARMS; do
                if [ "$arm" = "tabmda_encoder" ]; then
                    # ==== Grid search over in-context subsetting ====
                    for cs in $CONTEXT_SIZES; do
                        for nc in $NUM_CONTEXTS_GRID; do
                            TOTAL=$((TOTAL + 1))
                            run_one "$dataset" "$nrs" "$repeat" "$arm" "$cs" "$nc"
                        done
                    done
                else
                    TOTAL=$((TOTAL + 1))
                    run_one "$dataset" "$nrs" "$repeat" "$arm" "" ""
                fi
            done
        done
    done
done

log "done. total=$TOTAL  failed=$FAILED"
log "raw results: $RESULTS_FILE"
log "per-run logs: $LOGS_DIR"
echo

# ---------------------------------------------------------------------------
#  Aggregate over repeats -> mean +/- std
#  (tabmda_encoder: first select best grid point by mean val, then report it)
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" != "1" ] && [ -s "$RESULTS_FILE" ]; then
    log "aggregating over repeats (mean +/- sample std)..."
    python - "$RESULTS_FILE" <<'PY'
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

# Select best (context_size, num_contexts) per (dataset, n) by mean val accuracy.
chosen = {}   # (dataset, n) -> (val_mean, cs, nc)
for (dataset, n, arm, cs, nc), vals in rows.items():
    if arm != "tabmda_encoder":
        continue
    vm = sum(v[1] for v in vals) / len(vals)
    key = (dataset, n)
    if key not in chosen or vm > chosen[key][0]:
        chosen[key] = (vm, cs, nc)

order_arms = ["none", "smote", "tabmda_embedding", "tabmda_encoder"]
arm_rank = {a: i for i, a in enumerate(order_arms)}

display = []
for (dataset, n, arm, cs, nc), vals in rows.items():
    if arm == "tabmda_encoder":
        if chosen.get((dataset, n), (None, None, None))[1:] != (cs, nc):
            continue  # only report the selected grid point
        label = f"tabmda_encoder (cs={cs}, nc={nc})"
    else:
        label = arm
    display.append((dataset, n, arm, label, vals))

display.sort(key=lambda k: (k[0], k[1], arm_rank.get(k[2], 99)))

print()
print(f"{'dataset':<22}{'n':>5}  {'arm':<28}{'rep':>4}  {'train':>18}  {'val':>18}  {'test':>18}")
print("-" * 116)
for dataset, n, arm, label, vals in display:
    cnt = len(vals)
    tm, ts = meanstd([v[0] for v in vals])
    vm, vs = meanstd([v[1] for v in vals])
    em, es = meanstd([v[2] for v in vals])
    print(f"{dataset:<22}{n:>5}  {label:<28}{cnt:>4}  {tm:>9.4f}+-{ts:<7.4f}  {vm:>9.4f}+-{vs:<7.4f}  {em:>9.4f}+-{es:<7.4f}")
print()

# Summary of chosen TabMDA settings
if chosen:
    print("Selected TabMDA (tabmda_encoder) settings by best mean val balanced accuracy:")
    for (dataset, n), (vm, cs, nc) in sorted(chosen.items()):
        print(f"  {dataset:<22} n={n:>4}  ->  context_size={cs}  num_contexts={nc}  (mean val = {vm:.4f})")
print()
print("rep = number of successful repeats aggregated. std is sample std (ddof=1).")
print("tabmda_encoder is grid-searched over CONTEXT_SIZES x NUM_CONTEXTS_GRID; the best setting is chosen by mean val balanced accuracy.")
PY
    echo
fi

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
