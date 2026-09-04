#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
MIN_FREE_MB="${MIN_FREE_MB:-38000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
RAG_RETRIES="${RAG_RETRIES:-3}"
RAG_RETRY_SLEEP_SEC="${RAG_RETRY_SLEEP_SEC:-180}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/experiments/v6_hp_hyper_next/logs/resume_after_hf504_${STAMP}.log}"
ALL_COMMANDS="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/all_commands.log"
RAW_CSV="$ROOT_DIR/experiments/v6_hp_hyper_next/results/same_payload_multiseed_raw.csv"
TARGET_PAYLOAD="${TARGET_PAYLOAD:-0.070134}"
PAYLOAD_TOLERANCE="${PAYLOAD_TOLERANCE:-0.002}"
PERSIST_DIR="${PERSIST_DIR:-$ROOT_DIR/experiments/v6_hp_hyper_next/results/baseline_all1500_v3_rag_eval/ragtest_storage}"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-1500}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_FILE")" "$ROOT_DIR/experiments/v6_hp_hyper_next/results" "$ROOT_DIR/experiments/v6_hp_hyper_next/reports"

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

row_exists() {
  local method="$1"
  local seed="$2"
  local subset="$3"
  "$PYTHON_BIN" - "$RAW_CSV" "$method" "$seed" "$subset" <<'PY'
import csv
import sys

path, method, seed, subset = sys.argv[1:5]
try:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == method and row.get("seed") == seed and row.get("subset") == subset:
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PY
}

latest_run_dir() {
  local version_dir="$1"
  local experiment_name="$2"
  local suite_tag="$3"
  find "$version_dir/outputs/$experiment_name/$suite_tag" -name run_metadata.json -print 2>/dev/null \
    | sort \
    | tail -n 1 \
    | xargs dirname
}

latest_hf_model() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type d -name 'retriever_hf_*' -print | sort | tail -n 1
}

run_rag_eval_with_retry() {
  local gpu_id="$1"
  local model_dir="$2"
  local rag_dir="$3"
  local subset_path="$4"
  local per_query="$rag_dir/per_query.jsonl"
  mkdir -p "$rag_dir"
  local attempt=1
  while (( attempt <= RAG_RETRIES )); do
    log_msg "[rag-eval-attempt] $(date '+%F %T %Z') attempt=$attempt/$RAG_RETRIES rag_dir=$rag_dir"
    if /usr/bin/time -f "runtime_sec=%e" -o "$rag_dir/runtime.txt" \
      env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$ROOT_DIR/V6-HP1/run_rag_eval.py" \
        --model "$model_dir" \
        --script main_100_test.py \
        --output-dir "$rag_dir" \
        --python "$PYTHON_BIN" \
        --dataset hotpot_qa \
        --hotpot-split validation \
        --eval-num-examples "$EVAL_NUM_EXAMPLES" \
        --query-subset "$subset_path" \
        --ragtest-persist-dir "$PERSIST_DIR" \
        --save-per-query \
        --per-query-output "$per_query"; then
      return 0
    fi
    log_msg "[rag-eval-failed] $(date '+%F %T %Z') attempt=$attempt rag_dir=$rag_dir"
    attempt=$((attempt + 1))
    if (( attempt <= RAG_RETRIES )); then
      sleep "$RAG_RETRY_SLEEP_SEC"
    fi
  done
  return 1
}

collect_result() {
  local method="$1"
  local version="$2"
  local seed="$3"
  local subset_name="$4"
  local subset_path="$5"
  local run_dir="$6"
  local rag_dir="$7"
  local runtime=""
  if [[ -f "$rag_dir/runtime.txt" ]]; then
    runtime="$(sed 's/^runtime_sec=//' "$rag_dir/runtime.txt")"
  fi
  "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/collect_next_benchmark_result.py" \
    --csv "$RAW_CSV" \
    --method "$method" \
    --version "$version" \
    --seed "$seed" \
    --subset-name "$subset_name" \
    --subset-path "$subset_path" \
    --run-dir "$run_dir" \
    --rag-dir "$rag_dir" \
    --target-payload "$TARGET_PAYLOAD" \
    --payload-tolerance "$PAYLOAD_TOLERANCE" \
    --runtime-sec "$runtime"
}

run_upstream_case() {
  local gpu_id="$1"
  local method="$2"
  local version="$3"
  local script="$4"
  local strategy="$5"
  local task_name="$6"
  local seed="$7"
  local subset_name="$8"
  local experiment_name="$9"
  local suite_tag="${method}_${subset_name}_s${seed}"
  shift 9

  log_msg "[case-start] $(date '+%F %T %Z') method=$method subset=$subset_name seed=$seed"
  "$PYTHON_BIN" "$script" \
    --strategy "$strategy" \
    --topk 2 \
    --warmup 0 \
    --rounds 10 \
    --clients 5 \
    --epochs 1 \
    --batch-size 1 \
    --gpu "$gpu_id" \
    --seed "$seed" \
    --experiment-name "$experiment_name" \
    --suite-tag "$suite_tag" \
    --task-name "$task_name" \
    "$@"
}

run_method_case() {
  local method="$1"
  local version="$2"
  local seed="$3"
  local subset_name="$4"
  local experiment_name="$5"
  local output_root="$6"
  local subset_path="$ROOT_DIR/experiments/v6_hp_hyper_next/subsets/${subset_name}.json"

  if row_exists "$method" "$seed" "$subset_name"; then
    log_msg "[case-skip-existing] $(date '+%F %T %Z') method=$method subset=$subset_name seed=$seed"
    return 0
  fi

  local gpu_id
  gpu_id="$(wait_for_gpu "${method}_${subset_name}_s${seed}")"
  local suite_tag="${method}_${subset_name}_s${seed}"
  local version_dir="$ROOT_DIR/$version"
  local script="$version_dir/run_upstream.py"
  local run_dir=""
  local model_dir=""
  local rag_dir="$output_root/rag_eval/$suite_tag"

  case "$version" in
    V3)
      run_upstream_case "$gpu_id" "$method" "$version" "$script" "hypernet_v3" "num5_dir_a03_imb00_ts0_v3" "$seed" "$subset_name" "$experiment_name" \
        --score-mode value \
        --budget-mode fixed \
        --layerwise-budget
      ;;
    V4)
      run_upstream_case "$gpu_id" "$method" "$version" "$script" "hypernet_v4" "num5_dir_a03_imb00_ts0_v4" "$seed" "$subset_name" "$experiment_name" \
        --score-mode value \
        --budget-mode fixed \
        --disable-utility-memory \
        --layerwise-budget
      ;;
    V5)
      run_upstream_case "$gpu_id" "$method" "$version" "$script" "hypernet_v5" "num5_dir_a03_imb00_ts0_v5" "$seed" "$subset_name" "$experiment_name" \
        --score-mode value \
        --budget-mode fixed \
        --disable-utility-memory \
        --layerwise-budget
      ;;
    V6-HP1)
      run_upstream_case "$gpu_id" "$method" "$version" "$script" "hypernet_v6" "num5_dir_a03_imb00_ts0_v6hp1" "$seed" "$subset_name" "$experiment_name" \
        --score-mode value \
        --budget-mode fixed \
        --disable-utility-memory \
        --layerwise-budget \
        --hard-query-scale 0.8231144963873167 \
        --hard-client-threshold 0.754190270516828 \
        --adaptive-expand-threshold 0.7479361444551627 \
        --adaptive-shrink-threshold 0.489523383052344 \
        --utility-expand-threshold 1.576850439637794 \
        --rawdata-path "$ROOT_DIR/FedE/select_data_hotpot_train_5000.json" \
        --rag-dataset hotpot_qa \
        --rag-hotpot-split validation \
        --rag-hotpot-max-examples "$EVAL_NUM_EXAMPLES"
      ;;
    *)
      echo "[error] unsupported version=$version" >&2
      exit 2
      ;;
  esac

  run_dir="$(latest_run_dir "$version_dir" "$experiment_name" "$suite_tag")"
  model_dir="$(latest_hf_model "$run_dir")"
  run_rag_eval_with_retry "$gpu_id" "$model_dir" "$rag_dir" "$subset_path"
  collect_result "$method" "$version" "$seed" "$subset_name" "$subset_path" "$run_dir" "$rag_dir"
  log_msg "[case-done] $(date '+%F %T %Z') method=$method subset=$subset_name seed=$seed"
}

retry_failed_v3_hard1000_seed43() {
  local method="V3_topk2_fixed"
  local version="V3"
  local seed="43"
  local subset_name="hotpot_hard_1000"
  local subset_path="$ROOT_DIR/experiments/v6_hp_hyper_next/subsets/${subset_name}.json"
  local run_dir="$ROOT_DIR/V3/outputs/v6_hp_hyper_next_multiseed_hotpot_hard_1000_s43_20260629_135708/V3_topk2_fixed_hotpot_hard_1000_s43/num5_dir_a03_imb00_ts0_v3/hypernet-v3_k2_w0_s43_enc0_score-value_budget-fixed_hist5_client1_block1"
  local model_dir
  model_dir="$(latest_hf_model "$run_dir")"
  local rag_dir="$ROOT_DIR/experiments/v6_hp_hyper_next/results/same_payload_multiseed_hotpot_hard_1000_s43_20260629_135708/rag_eval/V3_topk2_fixed_hotpot_hard_1000_s43"

  if row_exists "$method" "$seed" "$subset_name"; then
    log_msg "[failed-retry-skip-existing] $(date '+%F %T %Z') method=$method subset=$subset_name seed=$seed"
    return 0
  fi
  if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
    echo "[error] failed retry model_dir missing under $run_dir" >&2
    exit 3
  fi
  local gpu_id
  gpu_id="$(wait_for_gpu "retry_${method}_${subset_name}_s${seed}")"
  log_msg "[failed-retry-start] $(date '+%F %T %Z') method=$method subset=$subset_name seed=$seed model=$model_dir"
  run_rag_eval_with_retry "$gpu_id" "$model_dir" "$rag_dir" "$subset_path"
  collect_result "$method" "$version" "$seed" "$subset_name" "$subset_path" "$run_dir" "$rag_dir"
  log_msg "[failed-retry-done] $(date '+%F %T %Z') method=$method subset=$subset_name seed=$seed"
}

run_remaining_multiseed() {
  local experiment_name="v6_hp_hyper_next_multiseed_resume_${STAMP}"
  for seed_subset in "43 hotpot_hard_1000" "43 hotpot_hard_500" "44 hotpot_all_1000" "44 hotpot_hard_1000" "44 hotpot_hard_500"; do
    local seed subset_name
    read -r seed subset_name <<<"$seed_subset"
    local output_root="$ROOT_DIR/experiments/v6_hp_hyper_next/results/same_payload_multiseed_${subset_name}_s${seed}_${STAMP}"
    run_method_case "V3_topk2_fixed" "V3" "$seed" "$subset_name" "$experiment_name" "$output_root"
    run_method_case "V4_topk2_fixed" "V4" "$seed" "$subset_name" "$experiment_name" "$output_root"
    run_method_case "V5_topk2_fixed" "V5" "$seed" "$subset_name" "$experiment_name" "$output_root"
    run_method_case "V6_HP_hyper_anchor" "V6-HP1" "$seed" "$subset_name" "$experiment_name" "$output_root"
  done
}

run_ablation_group() {
  local group="$1"
  local gpu_id
  gpu_id="$(wait_for_gpu "ablation_${group}")"
  log_msg "[ablation-start] $(date '+%F %T %Z') group=$group gpu=$gpu_id"
  PYTHON_BIN="$PYTHON_BIN" \
    GPU_ID="$gpu_id" \
    GROUP="$group" \
    STAMP="$STAMP" \
    EXPERIMENT_NAME="v6_hp_hyper_next_${group}_ablation_resume_${STAMP}" \
    OUTPUT_ROOT="$ROOT_DIR/experiments/v6_hp_hyper_next/results/${group}_ablation_resume_${STAMP}" \
    bash "$ROOT_DIR/experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh" 2>&1 | tee -a "$LOG_FILE"
  log_msg "[ablation-done] $(date '+%F %T %Z') group=$group"
}

write_launch_report() {
  local report="$ROOT_DIR/experiments/v6_hp_hyper_next/reports/resume_after_hf504_${STAMP}.md"
  {
    echo "# Resume After HF 504 Launch Report"
    echo
    echo "Date: $(date '+%F %T %Z')"
    echo
    echo "## Scope"
    echo
    echo "- Retry failed V3 downstream evaluation: \`hotpot_hard_1000 / seed=43\`."
    echo "- Append recovered row to \`same_payload_multiseed_raw.csv\`."
    echo "- Continue remaining multi-seed stages without rerunning completed B3 or \`all_1000 / seed=43\` rows."
    echo "- Then run ablation groups in order: \`pooler -> layerwise -> score_mode -> hard_weight -> adaptive\`."
    echo
    echo "## Runtime Controls"
    echo
    echo "- Stamp: \`$STAMP\`"
    echo "- Log: \`$LOG_FILE\`"
    echo "- GPU free-memory threshold: \`${MIN_FREE_MB} MiB\`"
    echo "- RAG eval retries: \`$RAG_RETRIES\`"
    echo "- Retry sleep: \`${RAG_RETRY_SLEEP_SEC}s\`"
  } > "$report"
  log_msg "[report-written] $report"
}

log_msg "[resume-start] $(date '+%F %T %Z') stamp=$STAMP"
write_launch_report
retry_failed_v3_hard1000_seed43
run_remaining_multiseed
for group in pooler layerwise score_mode hard_weight adaptive; do
  run_ablation_group "$group"
done
log_msg "[resume-done] $(date '+%F %T %Z') stamp=$STAMP"
