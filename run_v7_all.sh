#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MODE="${1:-first_pass}"
LOG="$ROOT/v7_all.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

run_suite() {
  local suite="$1"
  log "Starting upstream suite: $suite"
  "$PYTHON_BIN" "$ROOT/V7/run_experiment_suite.py" --suite "$suite" 2>&1 | tee -a "$LOG"
  log "Finished upstream suite: $suite"

  if [ -f "$ROOT/V7/run_all_rag_eval.py" ]; then
    log "Starting downstream RAG eval: $suite"
    "$PYTHON_BIN" "$ROOT/V7/run_all_rag_eval.py" \
      --upstream-root "$ROOT/V7/outputs/pprag_fl_v7/$suite" \
      --output-root "$ROOT/V7/outputs/rag_eval_all_v7/$suite" \
      2>&1 | tee -a "$LOG"
    log "Finished downstream RAG eval: $suite"
  fi

  if [ -f "$ROOT/V7/finalize_pipeline.py" ]; then
    log "Finalizing suite: $suite"
    "$PYTHON_BIN" "$ROOT/V7/finalize_pipeline.py" \
      --suite "$suite" \
      --upstream-root "$ROOT/V7/outputs/pprag_fl_v7/$suite" \
      --downstream-root "$ROOT/V7/outputs/rag_eval_all_v7/$suite" \
      2>&1 | tee -a "$LOG" || true
  fi
}

collect_and_analyze() {
  if [ -f "$ROOT/V7/collect_v7_results.py" ]; then
    log "Collecting V7 result summaries"
    "$PYTHON_BIN" "$ROOT/V7/collect_v7_results.py" 2>&1 | tee -a "$LOG" || true
  fi
  if [ -f "$ROOT/V7/write_v7_analysis.py" ]; then
    log "Writing V7 automatic analysis"
    "$PYTHON_BIN" "$ROOT/V7/write_v7_analysis.py" 2>&1 | tee -a "$LOG" || true
  fi
}

cd "$ROOT"
touch "$LOG"
log "V7 automation started with mode=$MODE"

case "$MODE" in
  smoke)
    SUITES=(smoke)
    ;;
  first_pass)
    SUITES=(v7_main v7_budget_aligned v7_hardquery)
    ;;
  full_pass)
    SUITES=(v7_main v7_budget_aligned v7_heterogeneity v7_hardquery v7_ablation_signal v7_ablation_agent_level v7_cost_efficiency v7_explain)
    ;;
  next)
    SUITES=(v7_heterogeneity v7_ablation_signal v7_hardquery_strong)
    ;;
  collect)
    collect_and_analyze
    log "V7 collection mode completed"
    exit 0
    ;;
  *)
    log "ERROR: unknown mode=$MODE"
    exit 2
    ;;
esac

for suite in "${SUITES[@]}"; do
  run_suite "$suite"
done

if [ -f "$ROOT/V7/finalize_pipeline.py" ]; then
  log "Running all_v7 finalize"
  "$PYTHON_BIN" "$ROOT/V7/finalize_pipeline.py" \
    --suite all_v7 \
    --upstream-root "$ROOT/V7/outputs/pprag_fl_v7" \
    --downstream-root "$ROOT/V7/outputs/rag_eval_all_v7" \
    2>&1 | tee -a "$LOG" || true
fi

collect_and_analyze
log "V7 automation completed"
