#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED_LIST="${SEED_LIST:-0,1,2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAG_SCRIPT="${RAG_SCRIPT:-main_100_test.py}"
FORCE_RAG="${FORCE_RAG:-0}"

UPSTREAM_ROOT="$ROOT_DIR/V5/outputs/pprag_fl_v5"
DOWNSTREAM_ROOT="$ROOT_DIR/V5/outputs/rag_eval_all_v5"
REPORT_ROOT="$ROOT_DIR/实验分析报告/V5"

ensure_finalize() {
  local suite_name="$1"
  local suite_upstream="$UPSTREAM_ROOT/$suite_name"
  local suite_downstream="$DOWNSTREAM_ROOT/$suite_name"
  if ls "$REPORT_ROOT"/full_pipeline_"$suite_name"_* >/dev/null 2>&1; then
    echo "[skip] full pipeline for $suite_name already exists"
    return 0
  fi
  local cmd=(
    "$PYTHON_BIN" V5/finalize_pipeline.py
    --suite-name "$suite_name"
    --upstream-root "$suite_upstream"
    --downstream-root "$suite_downstream"
    --script "$RAG_SCRIPT"
    --python "$PYTHON_BIN"
  )
  if [[ "$FORCE_RAG" == "1" ]]; then
    cmd+=(--force-rag)
  fi
  echo "[finalize] $suite_name"
  "${cmd[@]}"
}

run_suite() {
  local suite_name="$1"
  if ls "$REPORT_ROOT"/suite_"$suite_name"_* >/dev/null 2>&1; then
    echo "[skip] suite report for $suite_name already exists"
  else
    echo "[run] $suite_name"
    "$PYTHON_BIN" V5/run_experiment_suite.py \
      --suite "$suite_name" \
      --gpu "$GPU_ID" \
      --batch-size "$BATCH_SIZE" \
      --seed-list "$SEED_LIST"
  fi
  ensure_finalize "$suite_name"
}

run_suite v5_main
run_suite v5_budget
run_suite v5_budget_aligned
run_suite v5_heterogeneity
run_suite v5_hardquery
run_suite v5_ablation_signal
run_suite v5_ablation_budget
run_suite v5_explain

if ls "$REPORT_ROOT"/full_pipeline_all_v5_* >/dev/null 2>&1; then
  echo "[skip] full_pipeline_all_v5 already exists"
else
  cmd=(
    "$PYTHON_BIN" V5/finalize_pipeline.py
    --suite-name all_v5
    --upstream-root "$UPSTREAM_ROOT"
    --downstream-root "$DOWNSTREAM_ROOT"
    --script "$RAG_SCRIPT"
    --python "$PYTHON_BIN"
  )
  if [[ "$FORCE_RAG" == "1" ]]; then
    cmd+=(--force-rag)
  fi
  echo "[finalize] all_v5"
  "${cmd[@]}"
fi

echo "run_v5_all completed"
