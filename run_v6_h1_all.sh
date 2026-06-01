#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
ROUNDS="${ROUNDS:-25}"
CLIENTS="${CLIENTS:-5}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED_LIST="${SEED_LIST:-0,1,2}"
PARTITIONER="${PARTITIONER:-DirichletPartitioner}"
DIR_ALPHA="${DIR_ALPHA:-0.3}"
TASK_SEED="${TASK_SEED:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v6_h1}"
LOG_DIR="${LOG_DIR:-V6-H1/outputs/logs}"
DEFAULT_SUPV2_PY="$HOME/anaconda3/envs/supv2/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SUPV2_PY" ]]; then
    PYTHON_BIN="$DEFAULT_SUPV2_PY"
  else
    PYTHON_BIN="python3"
  fi
fi

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/v6_h1_all_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

run_suite() {
  local suite="$1"
  echo "[$(date '+%F %T')] START upstream suite: ${suite}"
  "$PYTHON_BIN" V6-H1/run_experiment_suite.py \
    --suite "$suite" \
    --rounds "$ROUNDS" \
    --clients "$CLIENTS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --gpu "$GPU" \
    --experiment-name "$EXPERIMENT_NAME" \
    --partitioner "$PARTITIONER" \
    --dir-alpha "$DIR_ALPHA" \
    --task-seed "$TASK_SEED" \
    --seed-list "$SEED_LIST"
  echo "[$(date '+%F %T')] DONE upstream suite: ${suite}"
}

finalize_suite() {
  local suite="$1"
  echo "[$(date '+%F %T')] START downstream full-set evaluation: ${suite}"
  "$PYTHON_BIN" V6-H1/finalize_pipeline.py \
    --suite-name "$suite" \
    --upstream-root "V6-H1/outputs/${EXPERIMENT_NAME}/${suite}" \
    --downstream-root "V6-H1/outputs/rag_eval_all_v6_h1/${suite}" \
    --script main_100_test.py \
    --python "$PYTHON_BIN" \
    --save-per-query
  echo "[$(date '+%F %T')] DONE downstream full-set evaluation: ${suite}"
}

finalize_hard_subset_suite() {
  local suite="$1"
  local subset_path="$2"
  echo "[$(date '+%F %T')] START downstream hard-query evaluation: ${suite}"
  "$PYTHON_BIN" V6-H1/finalize_pipeline.py \
    --suite-name "${suite}_stable_hardquery" \
    --upstream-root "V6-H1/outputs/${EXPERIMENT_NAME}/${suite}" \
    --downstream-root "V6-H1/outputs/rag_eval_hard_v6_h1/${suite}" \
    --script main_100_test.py \
    --python "$PYTHON_BIN" \
    --query-subset "$subset_path" \
    --save-per-query \
    --force-rag
  echo "[$(date '+%F %T')] DONE downstream hard-query evaluation: ${suite}"
}

echo "[$(date '+%F %T')] V6-H1 full automation starts"
echo "Log: ${MASTER_LOG}"

for suite in \
  v6h1_main \
  v6h1_budget_aligned \
  v6h1_heterogeneity \
  v6h1_hardquery \
  v6h1_ablation_signal \
  v6h1_ablation_budget \
  v6h1_explain
do
  run_suite "$suite"
  finalize_suite "$suite"
done

echo "[$(date '+%F %T')] Build stable hard-query subset from budget-aligned baseline"
"$PYTHON_BIN" V6-H1/build_hard_query_subset.py \
  --per-query-root "V6-H1/outputs/rag_eval_all_v6_h1/v6h1_budget_aligned" \
  --output-dir "V6-H1/hard_queries" \
  --baseline-pattern "hypernet-v3" \
  --min-hard-seeds 2 \
  --suite-name "v6h1_stable_hardquery"

HARD_SUBSET="V6-H1/hard_queries/stable_hard_queries.json"
for suite in v6h1_main v6h1_budget_aligned v6h1_hardquery v6h1_heterogeneity
do
  finalize_hard_subset_suite "$suite" "$HARD_SUBSET"
done

"$PYTHON_BIN" V6-H1/write_experiment_docs.py \
  --full-root "实验分析报告/V6-H1" \
  --output-record "V6-H1/v6_h1_complete_experiment_record_cn.md" \
  --output-analysis "V6-H1/v6_h1_complete_experiment_analysis_cn.md"

echo "[$(date '+%F %T')] V6-H1 full automation finished"
