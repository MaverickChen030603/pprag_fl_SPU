#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TARGET_PAYLOAD="${TARGET_PAYLOAD:-0.070134}"
PAYLOAD_TOLERANCE="${PAYLOAD_TOLERANCE:-0.002}"
SUBSET_NAME="${SUBSET_NAME:-hotpot_all_1000}"
SUBSET_PATH="${SUBSET_PATH:-$ROOT_DIR/experiments/v6_hp_hyper_next/subsets/${SUBSET_NAME}.json}"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-1500}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-v6_hp_hyper_next_same_payload_b1_$STAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/experiments/v6_hp_hyper_next/results/same_payload_baseline_b1_$STAMP}"
RAW_CSV="${RAW_CSV:-$ROOT_DIR/experiments/v6_hp_hyper_next/results/same_payload_baseline_raw.csv}"
COMMAND_LOG="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/same_payload_baseline_commands.log"
ALL_COMMANDS="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/all_commands.log"
PERSIST_DIR="${PERSIST_DIR:-$ROOT_DIR/experiments/v6_hp_hyper_next/results/baseline_all1500_v3_rag_eval/ragtest_storage}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RAW_CSV")" "$(dirname "$COMMAND_LOG")"

if [[ ! -f "$SUBSET_PATH" ]]; then
  echo "[error] subset not found: $SUBSET_PATH" >&2
  exit 2
fi

latest_run_dir() {
  local version_dir="$1"
  local suite_tag="$2"
  find "$version_dir/outputs/$EXPERIMENT_NAME/$suite_tag" -name run_metadata.json -print 2>/dev/null \
    | sort \
    | tail -n 1 \
    | xargs dirname
}

latest_hf_model() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type d -name 'retriever_hf_*' -print | sort | tail -n 1
}

log_msg() {
  local msg="$1"
  echo "$msg" | tee -a "$COMMAND_LOG" "$ALL_COMMANDS"
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
  local version="$2"
  local run_dir="$3"
  local rag_dir="$4"
  local runtime=""
  if [[ -f "$rag_dir/runtime.txt" ]]; then
    runtime="$(sed 's/^runtime_sec=//' "$rag_dir/runtime.txt")"
  fi
  "$PYTHON_BIN" "$ROOT_DIR/experiments/v6_hp_hyper_next/collect_next_benchmark_result.py" \
    --csv "$RAW_CSV" \
    --method "$method" \
    --version "$version" \
    --seed "$SEED" \
    --subset-name "$SUBSET_NAME" \
    --subset-path "$SUBSET_PATH" \
    --run-dir "$run_dir" \
    --rag-dir "$rag_dir" \
    --target-payload "$TARGET_PAYLOAD" \
    --payload-tolerance "$PAYLOAD_TOLERANCE" \
    --runtime-sec "$runtime"
}

run_case() {
  local method="$1"
  local version="$2"
  local script="$3"
  local strategy="$4"
  local task_name="$5"
  shift 5
  local suite_tag="${method}_${SUBSET_NAME}_s${SEED}"
  local version_dir="$ROOT_DIR/$version"
  local rag_dir="$OUTPUT_ROOT/rag_eval/$suite_tag"

  log_msg "[case-start] $(date '+%F %T %Z') method=$method version=$version subset=$SUBSET_NAME seed=$SEED"
  "$PYTHON_BIN" "$script" \
    --strategy "$strategy" \
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
    --task-name "$task_name" \
    "$@"

  local run_dir
  run_dir="$(latest_run_dir "$version_dir" "$suite_tag")"
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    echo "[error] cannot find run dir for $method" >&2
    exit 3
  fi
  local model_dir
  model_dir="$(latest_hf_model "$run_dir")"
  if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
    echo "[error] cannot find HF model under $run_dir" >&2
    exit 4
  fi

  run_rag_eval "$model_dir" "$rag_dir"
  collect_result "$method" "$version" "$run_dir" "$rag_dir"
  log_msg "[case-end] $(date '+%F %T %Z') method=$method rag_dir=$rag_dir"
}

log_msg "[start] $(date '+%F %T %Z') experiment=$EXPERIMENT_NAME subset=$SUBSET_NAME seed=$SEED target_payload=$TARGET_PAYLOAD tolerance=$PAYLOAD_TOLERANCE"

run_case "V3_topk2_fixed" "V3" "$ROOT_DIR/V3/run_upstream.py" "hypernet_v3" "num5_dir_a03_imb00_ts0_v3" \
  --score-mode value \
  --budget-mode fixed \
  --layerwise-budget

run_case "V4_topk2_fixed" "V4" "$ROOT_DIR/V4/run_upstream.py" "hypernet_v4" "num5_dir_a03_imb00_ts0_v4" \
  --score-mode value \
  --budget-mode fixed \
  --disable-utility-memory \
  --layerwise-budget

run_case "V5_topk2_fixed" "V5" "$ROOT_DIR/V5/run_upstream.py" "hypernet_v5" "num5_dir_a03_imb00_ts0_v5" \
  --score-mode value \
  --budget-mode fixed \
  --disable-utility-memory \
  --layerwise-budget

run_case "V6_HP_hyper_anchor" "V6-HP1" "$ROOT_DIR/V6-HP1/run_upstream.py" "hypernet_v6" "num5_dir_a03_imb00_ts0_v6hp1" \
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

log_msg "[done] $(date '+%F %T %Z') raw_csv=$RAW_CSV output_root=$OUTPUT_ROOT"
