#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SUBSET_NAME="${SUBSET_NAME:-hotpot_hard_1000}"
SUBSET_PATH="${SUBSET_PATH:-$ROOT_DIR/experiments/v6_hp_hyper_next/subsets/${SUBSET_NAME}.json}"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-1500}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-v6_hp_hyper_next_selection_ablation_$STAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/experiments/v6_hp_hyper_next/results/selection_diversity_ablation_$STAMP}"
RAW_CSV="${RAW_CSV:-}"
COMMAND_LOG="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/selection_diversity_ablation_commands.log"
ALL_COMMANDS="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/all_commands.log"
PERSIST_DIR="${PERSIST_DIR:-$ROOT_DIR/experiments/v6_hp_hyper_next/results/baseline_all1500_v3_rag_eval/ragtest_storage}"
GROUP="${GROUP:-all}"
if [[ -z "$RAW_CSV" ]]; then
  case "$GROUP" in
    pooler) RAW_CSV="$ROOT_DIR/experiments/v6_hp_hyper_next/results/pooler_ablation_raw.csv" ;;
    layerwise) RAW_CSV="$ROOT_DIR/experiments/v6_hp_hyper_next/results/layerwise_ablation_raw.csv" ;;
    score_mode) RAW_CSV="$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_mode_ablation_raw.csv" ;;
    hard_weight) RAW_CSV="$ROOT_DIR/experiments/v6_hp_hyper_next/results/hard_weight_ablation_raw.csv" ;;
    adaptive) RAW_CSV="$ROOT_DIR/experiments/v6_hp_hyper_next/results/adaptive_realloc_pilot_raw.csv" ;;
    *) RAW_CSV="$ROOT_DIR/experiments/v6_hp_hyper_next/results/selection_diversity_ablation_raw.csv" ;;
  esac
fi

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RAW_CSV")" "$(dirname "$COMMAND_LOG")"

log_msg() {
  local msg="$1"
  echo "$msg" | tee -a "$COMMAND_LOG" "$ALL_COMMANDS"
}

latest_run_dir() {
  local suite_tag="$1"
  find "$ROOT_DIR/V6-HP1/outputs/$EXPERIMENT_NAME/$suite_tag" -name run_metadata.json -print 2>/dev/null | sort | tail -n 1 | xargs dirname
}

latest_hf_model() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type d -name 'retriever_hf_*' -print | sort | tail -n 1
}

run_rag_eval() {
  local model_dir="$1"
  local rag_dir="$2"
  local per_query="$rag_dir/per_query.jsonl"
  mkdir -p "$rag_dir"
  /usr/bin/time -f "runtime_sec=%e" -o "$rag_dir/runtime.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$ROOT_DIR/V6-HP1/run_rag_eval.py" \
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
}

collect_result() {
  local method="$1"
  local run_dir="$2"
  local rag_dir="$3"
  local runtime=""
  if [[ -f "$rag_dir/runtime.txt" ]]; then
    runtime="$(sed 's/^runtime_sec=//' "$rag_dir/runtime.txt")"
  fi
  "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/collect_next_benchmark_result.py" \
    --csv "$RAW_CSV" \
    --method "$method" \
    --version "V6-HP1" \
    --seed "$SEED" \
    --subset-name "$SUBSET_NAME" \
    --subset-path "$SUBSET_PATH" \
    --run-dir "$run_dir" \
    --rag-dir "$rag_dir" \
    --target-payload 0.070134 \
    --payload-tolerance 0.002 \
    --runtime-sec "$runtime"
}

run_v6_case() {
  local method="$1"
  shift
  local suite_tag="${method}_${SUBSET_NAME}_s${SEED}"
  local rag_dir="$OUTPUT_ROOT/rag_eval/$suite_tag"
  log_msg "[ablation-start] $(date '+%F %T %Z') method=$method group=$GROUP"
  "$PYTHON_BIN" "$ROOT_DIR/V6-HP1/run_upstream.py" \
    --strategy hypernet_v6 \
    --topk 2 \
    --warmup 0 \
    --rounds "$ROUNDS" \
    --clients 5 \
    --epochs 1 \
    --batch-size "$BATCH_SIZE" \
    --gpu "$GPU_ID" \
    --seed "$SEED" \
    --experiment-name "$EXPERIMENT_NAME" \
    --suite-tag "$suite_tag" \
    --task-name "num5_dir_a03_imb00_ts0_v6hp1" \
    --budget-mode fixed \
    --disable-utility-memory \
    --hard-query-scale 0.8231144963873167 \
    --hard-client-threshold 0.754190270516828 \
    --adaptive-expand-threshold 0.7479361444551627 \
    --adaptive-shrink-threshold 0.489523383052344 \
    --utility-expand-threshold 1.576850439637794 \
    --rawdata-path "$ROOT_DIR/FedE/select_data_hotpot_train_5000.json" \
    --rag-dataset hotpot_qa \
    --rag-hotpot-split validation \
    --rag-hotpot-max-examples "$EVAL_NUM_EXAMPLES" \
    "$@"
  local run_dir
  run_dir="$(latest_run_dir "$suite_tag")"
  local model_dir
  model_dir="$(latest_hf_model "$run_dir")"
  run_rag_eval "$model_dir" "$rag_dir"
  collect_result "$method" "$run_dir" "$rag_dir"
  log_msg "[ablation-end] $(date '+%F %T %Z') method=$method run_dir=$run_dir"
}

run_group_layerwise() {
  run_v6_case "v6_anchor_layerwise_on" --score-mode value --layerwise-budget
  run_v6_case "v6_layerwise_off" --score-mode value
}

run_group_score_mode() {
  run_v6_case "v6_score_value" --score-mode value --layerwise-budget
  run_v6_case "v6_score_downstream_value" --score-mode downstream_value --layerwise-budget
  run_v6_case "v6_score_delta" --score-mode delta --layerwise-budget
  run_v6_case "v6_score_grad_norm" --score-mode grad_norm --layerwise-budget
}

run_group_hard_weight() {
  run_v6_case "v6_hard_weight_off" --score-mode value --layerwise-budget --disable-hard-query-weighting --hard-weight-alpha 0.0
  run_v6_case "v6_hard_weight_default" --score-mode value --layerwise-budget
  run_v6_case "v6_hard_weight_strong" --score-mode value --layerwise-budget --hard-weight-alpha 2.0
  run_v6_case "v6_hard_weight_very_strong" --score-mode value --layerwise-budget --hard-weight-alpha 4.0
}

run_group_pooler() {
  run_v6_case "v6_no_pooler_cap" --score-mode value --layerwise-budget
  run_v6_case "v6_pooler_cap_25" --score-mode value --layerwise-budget --pooler-cap-ratio 0.25
  run_v6_case "v6_pooler_cap_10" --score-mode value --layerwise-budget --pooler-cap-ratio 0.10
  run_v6_case "v6_pooler_exclude" --score-mode value --layerwise-budget --exclude-pooler
}

run_group_adaptive() {
  run_v6_case "v6_fixed_anchor" --score-mode value --layerwise-budget
  run_v6_case "v6_adaptive_realloc_same_payload" --score-mode value --layerwise-budget --budget-mode adaptive_realloc
}

log_msg "[ablation-suite-start] $(date '+%F %T %Z') experiment=$EXPERIMENT_NAME group=$GROUP subset=$SUBSET_NAME"
case "$GROUP" in
  layerwise) run_group_layerwise ;;
  score_mode) run_group_score_mode ;;
  hard_weight) run_group_hard_weight ;;
  pooler) run_group_pooler ;;
  adaptive) run_group_adaptive ;;
  all)
    run_group_layerwise
    run_group_score_mode
    run_group_hard_weight
    run_group_pooler
    run_group_adaptive
    ;;
  *)
    echo "[error] unknown GROUP=$GROUP" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/collect_score_logging.py" \
  --raw-csv "$RAW_CSV" \
  --output-jsonl "$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_logging_raw.jsonl" \
  --output-summary "$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_logging_summary.csv" \
  --output-report "$ROOT_DIR/experiments/v6_hp_hyper_next/reports/score_logging_report.md"

case "$GROUP" in
  pooler)
    "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/summarize_selection_ablation.py" \
      --metrics-csv "$RAW_CSV" \
      --output-summary "$ROOT_DIR/experiments/v6_hp_hyper_next/results/pooler_ablation_summary.csv" \
      --output-report "$ROOT_DIR/experiments/v6_hp_hyper_next/reports/pooler_ablation_report.md" \
      --anchor-method "v6_no_pooler_cap" \
      --title "Pooler Dominance Ablation Report"
    ;;
  layerwise)
    "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/summarize_selection_ablation.py" \
      --metrics-csv "$RAW_CSV" \
      --output-summary "$ROOT_DIR/experiments/v6_hp_hyper_next/results/layerwise_ablation_summary.csv" \
      --output-report "$ROOT_DIR/experiments/v6_hp_hyper_next/reports/layerwise_ablation_report.md" \
      --anchor-method "v6_anchor_layerwise_on" \
      --title "Layerwise Budget Ablation Report"
    ;;
  score_mode)
    "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/summarize_selection_ablation.py" \
      --metrics-csv "$RAW_CSV" \
      --output-summary "$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_mode_ablation_summary.csv" \
      --output-report "$ROOT_DIR/experiments/v6_hp_hyper_next/reports/score_mode_ablation_report.md" \
      --anchor-method "v6_score_value" \
      --title "Score Mode Ablation Report"
    ;;
esac

"$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/run_method_identity_audit_after_ablation.py" \
  --raw-csv "$RAW_CSV" \
  --score-log "$ROOT_DIR/experiments/v6_hp_hyper_next/results/score_logging_raw.jsonl" \
  --output-raw "$ROOT_DIR/experiments/v6_hp_hyper_next/results/method_identity_audit_after_ablation_raw.jsonl" \
  --output-summary "$ROOT_DIR/experiments/v6_hp_hyper_next/results/method_identity_audit_after_ablation_summary.csv" \
  --output-report "$ROOT_DIR/experiments/v6_hp_hyper_next/reports/method_identity_audit_after_ablation_report.md"

log_msg "[ablation-suite-done] $(date '+%F %T %Z') raw_csv=$RAW_CSV output_root=$OUTPUT_ROOT"
