#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MODE="${1:-full}"
LOG="${LOG:-$ROOT/v7_h1_all.log}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v7_h1}"
ROUNDS="${H1_ROUNDS:-12}"
CLIENTS="${H1_CLIENTS:-5}"
EPOCHS="${H1_EPOCHS:-1}"
BATCH_SIZE="${H1_BATCH_SIZE:-8}"
LR="${H1_LR:-1e-5}"
GPU="${H1_GPU:-0}"
RUN_STANDARD_RAG="${RUN_STANDARD_RAG:-0}"
SEED_LIST="${H1_SEED_LIST:-0,1,2}"

mkdir -p "$ROOT/V7-H1/outputs" "$ROOT/实验分析报告/V7-H1"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_suite() {
  local suite="$1"
  log "START suite=$suite rounds=$ROUNDS seeds=$SEED_LIST"
  "$PYTHON_BIN" "$ROOT/V7-H1/run_experiment_suite.py" \
    --suite "$suite" \
    --experiment-name "$EXPERIMENT_NAME" \
    --rounds "$ROUNDS" \
    --clients "$CLIENTS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --gpu "$GPU" \
    --seed-list "$SEED_LIST" 2>&1 | tee -a "$LOG"

  local suite_root="$ROOT/V7-H1/outputs/$EXPERIMENT_NAME/$suite"
  log "SUMMARIZE suite=$suite"
  "$PYTHON_BIN" "$ROOT/V7-H1/summarize_results.py" \
    --root "$suite_root" \
    --output "$suite_root/summary" 2>&1 | tee -a "$LOG" || true

  log "STRICT_EVAL suite=$suite"
  "$PYTHON_BIN" "$ROOT/V7-H1/run_h1_strict_eval.py" \
    --upstream-root "$suite_root" \
    --output-root "$ROOT/V7-H1/outputs/h1_strict_eval/$suite" \
    --force 2>&1 | tee -a "$LOG"

  if [[ "$RUN_STANDARD_RAG" == "1" ]]; then
    log "STANDARD_RAG suite=$suite"
    "$PYTHON_BIN" "$ROOT/V7-H1/run_all_rag_eval.py" \
      --upstream-root "$suite_root" \
      --output-root "$ROOT/V7-H1/outputs/rag_eval_all_v7_h1/$suite" \
      --force 2>&1 | tee -a "$LOG"
  fi
  log "DONE suite=$suite"
}

case "$MODE" in
  smoke)
    SUITES=(smoke)
    ROUNDS="${H1_ROUNDS:-1}"
    SEED_LIST="${H1_SEED_LIST:-0}"
    ;;
  full)
    SUITES=(h1_hardquery_non_saturated h1_rare_domain_tail h1_action_space h1_ablation)
    ;;
  hardquery)
    SUITES=(h1_hardquery_non_saturated)
    ;;
  rare)
    SUITES=(h1_rare_domain_tail)
    ;;
  action)
    SUITES=(h1_action_space)
    ;;
  ablation)
    SUITES=(h1_ablation)
    ;;
  *)
    echo "Usage: $0 {smoke|full|hardquery|rare|action|ablation}" >&2
    exit 2
    ;;
esac

log "V7-H1 pipeline mode=$MODE root=$ROOT"
for suite in "${SUITES[@]}"; do
  run_suite "$suite"
done

log "GLOBAL_SUMMARY"
"$PYTHON_BIN" "$ROOT/V7-H1/summarize_results.py" \
  --root "$ROOT/V7-H1/outputs/$EXPERIMENT_NAME" \
  --output "$ROOT/V7-H1/outputs/$EXPERIMENT_NAME/summary" 2>&1 | tee -a "$LOG" || true

"$PYTHON_BIN" "$ROOT/V7-H1/run_h1_strict_eval.py" \
  --upstream-root "$ROOT/V7-H1/outputs/$EXPERIMENT_NAME" \
  --output-root "$ROOT/V7-H1/outputs/h1_strict_eval/all_h1" \
  --force 2>&1 | tee -a "$LOG" || true

log "WRITE_REPORT"
"$PYTHON_BIN" "$ROOT/V7-H1/write_h1_analysis.py" 2>&1 | tee -a "$LOG"
log "V7-H1 pipeline complete mode=$MODE"
