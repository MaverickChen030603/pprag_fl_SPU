#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
MIN_FREE_MB="${MIN_FREE_MB:-38000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/experiments/v6_hp_hyper_next/logs/ablation_resume_from_score_grad_norm_${STAMP}.log}"
ALL_COMMANDS="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/all_commands.log"
PERSIST_DIR="${PERSIST_DIR:-$ROOT_DIR/experiments/v6_hp_hyper_next/results/baseline_all1500_v3_rag_eval/ragtest_storage}"
SUBSET_NAME="hotpot_hard_1000"
SUBSET_PATH="$ROOT_DIR/experiments/v6_hp_hyper_next/subsets/${SUBSET_NAME}.json"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-1500}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

log_msg() {
  local msg="$1"
  echo "$msg" | tee -a "$LOG_FILE" "$ALL_COMMANDS" >&2
}

wait_for_gpu() {
  local stage="$1"
  while true; do
    log_msg "[gpu-wait] $(date '+%F %T %Z') stage=$stage min_free_mb=$MIN_FREE_MB"
    local candidate
    candidate="$(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, -v min_free="$MIN_FREE_MB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 >= min_free) {print $1; exit}}'
    )"
    if [[ -n "${candidate:-}" ]]; then
      log_msg "[gpu-selected] $(date '+%F %T %Z') stage=$stage gpu=$candidate"
      printf '%s' "$candidate"
      return 0
    fi
    sleep "$CHECK_INTERVAL_SEC"
  done
}

csv_has_method() {
  local csv="$1"
  local method="$2"
  "$PYTHON_BIN" - "$csv" "$method" <<'PY'
import csv
import sys
path, method = sys.argv[1:3]
try:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == method:
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PY
}

latest_hf_model() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type d -name 'retriever_hf_*' -print | sort | tail -n 1
}

run_grad_norm_eval_if_missing() {
  local method="v6_score_grad_norm"
  local raw_csv="$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_mode_ablation_raw.csv"
  if csv_has_method "$raw_csv" "$method"; then
    log_msg "[grad-norm-skip-existing] $(date '+%F %T %Z') method=$method"
    return 0
  fi

  local run_dir="$ROOT_DIR/V6-HP1/outputs/v6_hp_hyper_next_score_mode_ablation_resume_from_layerwise_20260706_131022/v6_score_grad_norm_hotpot_hard_1000_s42/num5_dir_a03_imb00_ts0_v6hp1/hypernet-v6_k2_w0_s42_enc0_score-grad_norm_budget-fixed_hist5_client1_block1_hard1_util0"
  local model_dir
  model_dir="$(latest_hf_model "$run_dir")"
  if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
    echo "[error] missing grad_norm model under $run_dir" >&2
    exit 3
  fi

  local gpu_id
  gpu_id="$(wait_for_gpu "score_grad_norm_eval")"
  local rag_dir="$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_mode_ablation_resume_from_layerwise_20260706_131022/rag_eval/v6_score_grad_norm_hotpot_hard_1000_s42"
  local per_query="$rag_dir/per_query.jsonl"
  mkdir -p "$rag_dir"
  log_msg "[grad-norm-rag-start] $(date '+%F %T %Z') model=$model_dir"
  /usr/bin/time -f "runtime_sec=%e" -o "$rag_dir/runtime.txt" \
    env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$ROOT_DIR/V6-HP1/run_rag_eval.py" \
      --model "$model_dir" \
      --script main_100_test.py \
      --output-dir "$rag_dir" \
      --python "$PYTHON_BIN" \
      --dataset hotpot_qa \
      --hotpot-split validation \
      --eval-num-examples "$EVAL_NUM_EXAMPLES" \
      --query-subset "$SUBSET_PATH" \
      --ragtest-persist-dir "$PERSIST_DIR" \
      --save-per-query \
      --per-query-output "$per_query"

  local runtime=""
  if [[ -f "$rag_dir/runtime.txt" ]]; then
    runtime="$(sed 's/^runtime_sec=//' "$rag_dir/runtime.txt")"
  fi
  "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/collect_next_benchmark_result.py" \
    --csv "$raw_csv" \
    --method "$method" \
    --version "V6-HP1" \
    --seed 42 \
    --subset-name "$SUBSET_NAME" \
    --subset-path "$SUBSET_PATH" \
    --run-dir "$run_dir" \
    --rag-dir "$rag_dir" \
    --target-payload 0.070134 \
    --payload-tolerance 0.002 \
    --runtime-sec "$runtime"
  log_msg "[grad-norm-rag-done] $(date '+%F %T %Z') method=$method"
}

summarize_score_mode() {
  "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/summarize_selection_ablation.py" \
    --metrics-csv "$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_mode_ablation_raw.csv" \
    --output-summary "$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_mode_ablation_summary.csv" \
    --output-report "$ROOT_DIR/experiments/v6_hp_hyper_next/reports/score_mode_ablation_report.md" \
    --anchor-method "v6_score_value" \
    --title "Score Mode Ablation Report"
}

run_group() {
  local group="$1"
  local gpu_id
  gpu_id="$(wait_for_gpu "ablation_${group}")"
  log_msg "[ablation-resume-start] $(date '+%F %T %Z') group=$group gpu=$gpu_id"
  GPU_ID="$gpu_id" \
    GROUP="$group" \
    STAMP="$STAMP" \
    EXPERIMENT_NAME="v6_hp_hyper_next_${group}_ablation_resume_from_score_grad_norm_${STAMP}" \
    OUTPUT_ROOT="$ROOT_DIR/experiments/v6_hp_hyper_next/results/${group}_ablation_resume_from_score_grad_norm_${STAMP}" \
    bash "$ROOT_DIR/experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh" 2>&1 | tee -a "$LOG_FILE"
  log_msg "[ablation-resume-done] $(date '+%F %T %Z') group=$group"
}

log_msg "[resume-from-score-grad-norm-start] $(date '+%F %T %Z') stamp=$STAMP"
run_grad_norm_eval_if_missing
summarize_score_mode
for group in hard_weight adaptive; do
  run_group "$group"
done
log_msg "[resume-from-score-grad-norm-done] $(date '+%F %T %Z') stamp=$STAMP"
