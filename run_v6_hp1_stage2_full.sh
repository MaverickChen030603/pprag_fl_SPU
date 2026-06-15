#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
MIN_FREE_MB="${MIN_FREE_MB:-38000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
N_TRIALS="${N_TRIALS:-24}"
ROUNDS="${ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HARD_EVAL_EXAMPLES="${HARD_EVAL_EXAMPLES:-1500}"
HARD_TARGET_SIZE="${HARD_TARGET_SIZE:-800}"
STAGE2_EVAL_EXAMPLES="${STAGE2_EVAL_EXAMPLES:-1000}"
PAYLOAD_PENALTY_MODE="${PAYLOAD_PENALTY_MODE:-search}"
HARD_QUERY_SUBSET="${HARD_QUERY_SUBSET:-$ROOT_DIR/V6-HP1/data/hotpot_hard_query_subset.json}"
BEST_MODEL="${BEST_MODEL:-}"

cd "$ROOT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" >&2
}

select_gpu() {
  local busy_uuids
  busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | sort -u || true)"
  nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v min_free="$MIN_FREE_MB" -v busy_uuids="$busy_uuids" '
      {
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if (index("\n" busy_uuids "\n", "\n" $2 "\n") > 0) {
          next;
        }
        if ($3 >= min_free && $3 > best_free) {
          best_gpu = $1;
          best_free = $3;
        }
      }
      END {
        if (best_gpu != "") {
          print best_gpu;
        }
      }'
}

wait_for_gpu() {
  local gpu_id=""
  while [[ -z "$gpu_id" ]]; do
    gpu_id="$(select_gpu || true)"
    if [[ -n "$gpu_id" ]]; then
      echo "$gpu_id"
      return 0
    fi
    log "No GPU has >= ${MIN_FREE_MB} MiB free; retrying in ${GPU_POLL_SECONDS}s."
    sleep "$GPU_POLL_SECONDS"
  done
}

find_best_model() {
  if [[ -n "$BEST_MODEL" ]]; then
    echo "$BEST_MODEL"
    return 0
  fi
  find "$ROOT_DIR/V6-HP1/outputs/pprag_fl_v6_hp1_optuna/v6hp1_optuna_t0002" \
    -maxdepth 4 -type d -name 'retriever_hf_*' 2>/dev/null \
    | sort \
    | tail -n 1
}

GPU_ID="$(wait_for_gpu)"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
log "Selected physical GPU ${GPU_ID}; inside CUDA_VISIBLE_DEVICES, training uses --gpu 0."

if [[ ! -f "$HARD_QUERY_SUBSET" ]]; then
  BASELINE_MODEL="$(find_best_model)"
  if [[ -z "$BASELINE_MODEL" || ! -d "$BASELINE_MODEL" ]]; then
    log "Cannot find baseline model for hard subset generation."
    exit 2
  fi
  log "Building hard query subset from baseline model: $BASELINE_MODEL"
  "$PYTHON_BIN" "$ROOT_DIR/V6-HP1/build_hotpot_hard_subset.py" \
    --baseline-model "$BASELINE_MODEL" \
    --python "$PYTHON_BIN" \
    --eval-num-examples "$HARD_EVAL_EXAMPLES" \
    --target-size "$HARD_TARGET_SIZE" \
    --output "$HARD_QUERY_SUBSET"
else
  log "Hard query subset already exists: $HARD_QUERY_SUBSET"
fi

log "Starting V6-HP1 Optuna Stage2 search."
env \
  PYTHON_BIN="$PYTHON_BIN" \
  GPU_ID=0 \
  N_TRIALS="$N_TRIALS" \
  ROUNDS="$ROUNDS" \
  BATCH_SIZE="$BATCH_SIZE" \
  EVAL_SUBSET_TYPE=hard_only \
  EVAL_NUM_EXAMPLES="$STAGE2_EVAL_EXAMPLES" \
  HARD_QUERY_SUBSET="$HARD_QUERY_SUBSET" \
  PAYLOAD_PENALTY_MODE="$PAYLOAD_PENALTY_MODE" \
  "$ROOT_DIR/run_v6_hp1_optuna_stage2.sh"

log "V6-HP1 Stage2 full workflow completed."
