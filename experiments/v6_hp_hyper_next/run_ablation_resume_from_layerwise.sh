#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MIN_FREE_MB="${MIN_FREE_MB:-38000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/experiments/v6_hp_hyper_next/logs/ablation_resume_from_layerwise_${STAMP}.log}"
ALL_COMMANDS="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/all_commands.log"

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

run_group() {
  local group="$1"
  local gpu_id
  gpu_id="$(wait_for_gpu "ablation_${group}")"
  log_msg "[ablation-resume-start] $(date '+%F %T %Z') group=$group gpu=$gpu_id"
  GPU_ID="$gpu_id" \
    GROUP="$group" \
    STAMP="$STAMP" \
    EXPERIMENT_NAME="v6_hp_hyper_next_${group}_ablation_resume_from_layerwise_${STAMP}" \
    OUTPUT_ROOT="$ROOT_DIR/experiments/v6_hp_hyper_next/results/${group}_ablation_resume_from_layerwise_${STAMP}" \
    bash "$ROOT_DIR/experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh" 2>&1 | tee -a "$LOG_FILE"
  log_msg "[ablation-resume-done] $(date '+%F %T %Z') group=$group"
}

log_msg "[resume-from-layerwise-start] $(date '+%F %T %Z') stamp=$STAMP"

for group in layerwise score_mode hard_weight adaptive; do
  run_group "$group"
done

log_msg "[resume-from-layerwise-done] $(date '+%F %T %Z') stamp=$STAMP"
