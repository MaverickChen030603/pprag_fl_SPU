#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
ROUNDS="${ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_SUBSET_TYPE="${EVAL_SUBSET_TYPE:-all}"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-1000}"
HARD_QUERY_SUBSET="${HARD_QUERY_SUBSET:-$ROOT_DIR/V6-HP1/data/hotpot_hard_query_subset.json}"
STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-same_payload_baseline_hotpot_$STAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/V6-HP1-OPTUNA/outputs/baseline_same_payload_$STAMP}"
RESULT_CSV="$OUTPUT_ROOT/same_payload_baseline_results.csv"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_ROOT"

if [[ "$EVAL_SUBSET_TYPE" == "hard_only" && ! -f "$HARD_QUERY_SUBSET" ]]; then
  echo "[error] hard query subset not found: $HARD_QUERY_SUBSET" >&2
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

run_rag_eval() {
  local model_dir="$1"
  local rag_dir="$2"
  local cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/V6-HP1/run_rag_eval.py"
    --model "$model_dir"
    --script main_100_test.py
    --output-dir "$rag_dir"
    --python "$PYTHON_BIN"
    --dataset hotpot_qa
    --hotpot-split validation
    --eval-subset-type "$EVAL_SUBSET_TYPE"
    --eval-num-examples "$EVAL_NUM_EXAMPLES"
  )
  if [[ "$EVAL_SUBSET_TYPE" == "hard_only" ]]; then
    cmd+=(--hard-query-subset "$HARD_QUERY_SUBSET")
  fi
  "${cmd[@]}"
}

collect_result() {
  local label="$1"
  local version="$2"
  local run_dir="$3"
  local rag_dir="$4"
  "$PYTHON_BIN" "$ROOT_DIR/V6-HP1-OPTUNA/collect_benchmark_result.py" \
    --csv "$RESULT_CSV" \
    --label "$label" \
    --version "$version" \
    --run-dir "$run_dir" \
    --rag-dir "$rag_dir" \
    --eval-subset-type "$EVAL_SUBSET_TYPE" \
    --eval-num-examples "$EVAL_NUM_EXAMPLES"
}

run_case() {
  local label="$1"
  local version="$2"
  local script="$3"
  local strategy="$4"
  local task_name="$5"
  shift 5
  local suite_tag="${label}"
  local version_dir="$ROOT_DIR/$version"
  local rag_dir="$OUTPUT_ROOT/rag_eval/$label"

  echo "[case] $label version=$version strategy=$strategy"
  "$PYTHON_BIN" "$script" \
    --strategy "$strategy" \
    --topk 2 \
    --warmup 0 \
    --rounds "$ROUNDS" \
    --clients 5 \
    --epochs 1 \
    --batch-size "$BATCH_SIZE" \
    --gpu "$GPU_ID" \
    --seed 0 \
    --experiment-name "$EXPERIMENT_NAME" \
    --suite-tag "$suite_tag" \
    --task-name "$task_name" \
    "$@"

  local run_dir
  run_dir="$(latest_run_dir "$version_dir" "$suite_tag")"
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    echo "[error] cannot find run dir for $label" >&2
    exit 3
  fi
  local model_dir
  model_dir="$(latest_hf_model "$run_dir")"
  if [[ -z "$model_dir" || ! -d "$model_dir" ]]; then
    echo "[error] cannot find HF model under $run_dir" >&2
    exit 4
  fi

  run_rag_eval "$model_dir" "$rag_dir"
  collect_result "$label" "$version" "$run_dir" "$rag_dir"
}

run_case "v3_topk2_fixed" "V3" "$ROOT_DIR/V3/run_upstream.py" "hypernet_v3" "num5_dir_a03_imb00_ts0_v3" \
  --score-mode value \
  --budget-mode fixed \
  --layerwise-budget

run_case "v4_topk2_fixed" "V4" "$ROOT_DIR/V4/run_upstream.py" "hypernet_v4" "num5_dir_a03_imb00_ts0_v4" \
  --score-mode value \
  --budget-mode fixed \
  --disable-utility-memory \
  --layerwise-budget

run_case "v5_topk2_fixed" "V5" "$ROOT_DIR/V5/run_upstream.py" "hypernet_v5" "num5_dir_a03_imb00_ts0_v5" \
  --score-mode value \
  --budget-mode fixed \
  --disable-utility-memory \
  --layerwise-budget

run_case "v6_hp1_optuna_best" "V6-HP1" "$ROOT_DIR/V6-HP1/run_upstream.py" "hypernet_v6" "num5_dir_a03_imb00_ts0_v6hp1" \
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

echo "[done] wrote $RESULT_CSV"
