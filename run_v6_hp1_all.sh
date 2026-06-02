#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED_LIST="${SEED_LIST:-0,1,2}"
DEFAULT_SUPV2_PY="$HOME/anaconda3/envs/supv2/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SUPV2_PY" ]]; then
    PYTHON_BIN="$DEFAULT_SUPV2_PY"
  else
    PYTHON_BIN="python3"
  fi
fi

HOTPOT_TRAIN_EXAMPLES="${HOTPOT_TRAIN_EXAMPLES:-5000}"
HOTPOT_EVAL_SPLIT="${HOTPOT_EVAL_SPLIT:-validation}"
HOTPOT_EVAL_EXAMPLES="${HOTPOT_EVAL_EXAMPLES:-1000}"
RAG_SCRIPT="${RAG_SCRIPT:-main_100_test.py}"
FORCE_RAG="${FORCE_RAG:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v6_hp1}"
RAWDATA_PATH="${RAWDATA_PATH:-$ROOT_DIR/FedE/select_data_hotpot_train_${HOTPOT_TRAIN_EXAMPLES}.json}"
UPSTREAM_ROOT="$ROOT_DIR/V6-HP1/outputs/${EXPERIMENT_NAME}"
DOWNSTREAM_ROOT="$ROOT_DIR/V6-HP1/outputs/rag_eval_all_v6_hp1"
REPORT_ROOT="$ROOT_DIR/实验分析报告/V6-HP1"
LOG_DIR="$ROOT_DIR/V6-HP1/outputs/logs"

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/v6_hp1_all_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[env] ROOT_DIR=$ROOT_DIR"
echo "[env] PYTHON_BIN=$PYTHON_BIN"
echo "[env] GPU_ID=$GPU_ID BATCH_SIZE=$BATCH_SIZE SEED_LIST=$SEED_LIST"
echo "[env] HOTPOT_TRAIN_EXAMPLES=$HOTPOT_TRAIN_EXAMPLES HOTPOT_EVAL_SPLIT=$HOTPOT_EVAL_SPLIT HOTPOT_EVAL_EXAMPLES=$HOTPOT_EVAL_EXAMPLES"
echo "[env] RAWDATA_PATH=$RAWDATA_PATH"

echo "[prep] build hotpot upstream training data"
"$PYTHON_BIN" V6-HP1/prepare_hotpot_data.py \
  --split train \
  --max-examples "$HOTPOT_TRAIN_EXAMPLES" \
  --output "$RAWDATA_PATH"

ensure_finalize() {
  local suite_name="$1"
  local suite_upstream="$UPSTREAM_ROOT/$suite_name"
  local suite_downstream="$DOWNSTREAM_ROOT/$suite_name"
  if ls "$REPORT_ROOT"/full_pipeline_"$suite_name"_* >/dev/null 2>&1; then
    echo "[skip] full pipeline for $suite_name already exists"
    return 0
  fi
  local cmd=(
    "$PYTHON_BIN" V6-HP1/finalize_pipeline.py
    --suite-name "$suite_name"
    --upstream-root "$suite_upstream"
    --downstream-root "$suite_downstream"
    --script "$RAG_SCRIPT"
    --python "$PYTHON_BIN"
    --dataset hotpot_qa
    --hotpot-split "$HOTPOT_EVAL_SPLIT"
    --hotpot-max-examples "$HOTPOT_EVAL_EXAMPLES"
    --save-per-query
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
    "$PYTHON_BIN" V6-HP1/run_experiment_suite.py \
      --suite "$suite_name" \
      --gpu "$GPU_ID" \
      --batch-size "$BATCH_SIZE" \
      --seed-list "$SEED_LIST" \
      --experiment-name "$EXPERIMENT_NAME" \
      --rawdata-path "$RAWDATA_PATH" \
      --rag-dataset hotpot_qa \
      --rag-hotpot-split "$HOTPOT_EVAL_SPLIT" \
      --rag-hotpot-max-examples "$HOTPOT_EVAL_EXAMPLES"
  fi
  ensure_finalize "$suite_name"
}

run_suite v6hp1_main
run_suite v6hp1_budget_aligned
run_suite v6hp1_heterogeneity
run_suite v6hp1_hardquery
run_suite v6hp1_ablation_signal
run_suite v6hp1_ablation_budget
run_suite v6hp1_explain

if ls "$REPORT_ROOT"/full_pipeline_all_v6_hp1_* >/dev/null 2>&1; then
  echo "[skip] full_pipeline_all_v6_hp1 already exists"
else
  cmd=(
    "$PYTHON_BIN" V6-HP1/finalize_pipeline.py
    --suite-name all_v6_hp1
    --upstream-root "$UPSTREAM_ROOT"
    --downstream-root "$DOWNSTREAM_ROOT"
    --script "$RAG_SCRIPT"
    --python "$PYTHON_BIN"
    --dataset hotpot_qa
    --hotpot-split "$HOTPOT_EVAL_SPLIT"
    --hotpot-max-examples "$HOTPOT_EVAL_EXAMPLES"
    --save-per-query
  )
  if [[ "$FORCE_RAG" == "1" ]]; then
    cmd+=(--force-rag)
  fi
  echo "[finalize] all_v6_hp1"
  "${cmd[@]}"
fi

"$PYTHON_BIN" V6-HP1/write_experiment_docs.py \
  --full-root "$REPORT_ROOT" \
  --output-record "$ROOT_DIR/V6-HP1/v6_hp1_complete_experiment_record_cn.md" \
  --output-analysis "$ROOT_DIR/V6-HP1/v6_hp1_complete_experiment_analysis_cn.md"

echo "run_v6_hp1_all completed"
