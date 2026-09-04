#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
MIN_FREE_MB="${MIN_FREE_MB:-38000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
QUEUE_LOG="${QUEUE_LOG:-$ROOT_DIR/experiments/v6_hp_hyper_next/logs/b3_multiseed_ablation_sequence_${STAMP}.log}"
ALL_COMMANDS="$ROOT_DIR/experiments/v6_hp_hyper_next/logs/all_commands.log"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$QUEUE_LOG")" "$ROOT_DIR/experiments/v6_hp_hyper_next/results" "$ROOT_DIR/experiments/v6_hp_hyper_next/reports"

log_msg() {
  local msg="$1"
  echo "$msg" | tee -a "$QUEUE_LOG" "$ALL_COMMANDS" >&2
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

run_same_payload_stage() {
  local stage="$1"
  local subset_name="$2"
  local seed="$3"
  local raw_csv="$4"
  local experiment_prefix="$5"
  local output_prefix="$6"

  local gpu_id
  gpu_id="$(wait_for_gpu "$stage")"
  log_msg "[stage-start] $(date '+%F %T %Z') stage=$stage subset=$subset_name seed=$seed gpu=$gpu_id"
  PYTHON_BIN="$PYTHON_BIN" \
    GPU_ID="$gpu_id" \
    SEED="$seed" \
    SUBSET_NAME="$subset_name" \
    RAW_CSV="$raw_csv" \
    EXPERIMENT_NAME="${experiment_prefix}_${subset_name}_s${seed}_${STAMP}" \
    OUTPUT_ROOT="$ROOT_DIR/experiments/v6_hp_hyper_next/results/${output_prefix}_${subset_name}_s${seed}_${STAMP}" \
    bash "$ROOT_DIR/experiments/v6_hp_hyper_next/run_same_payload_baseline_b1.sh" 2>&1 | tee -a "$QUEUE_LOG"
  log_msg "[stage-done] $(date '+%F %T %Z') stage=$stage subset=$subset_name seed=$seed"
}

run_ablation_stage() {
  local group="$1"
  local gpu_id
  gpu_id="$(wait_for_gpu "ablation_${group}")"
  log_msg "[ablation-start] $(date '+%F %T %Z') group=$group gpu=$gpu_id"
  PYTHON_BIN="$PYTHON_BIN" \
    GPU_ID="$gpu_id" \
    GROUP="$group" \
    STAMP="$STAMP" \
    EXPERIMENT_NAME="v6_hp_hyper_next_${group}_ablation_${STAMP}" \
    OUTPUT_ROOT="$ROOT_DIR/experiments/v6_hp_hyper_next/results/${group}_ablation_${STAMP}" \
    bash "$ROOT_DIR/experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh" 2>&1 | tee -a "$QUEUE_LOG"
  log_msg "[ablation-done] $(date '+%F %T %Z') group=$group"
}

write_sequence_report() {
  local report="$ROOT_DIR/experiments/v6_hp_hyper_next/reports/b3_multiseed_ablation_sequence_${STAMP}.md"
  {
    echo "# B3 / Multi-seed / Ablation Sequence Launch Report"
    echo
    echo "Date: $(date '+%F %T %Z')"
    echo
    echo "## Launch Summary"
    echo
    echo "- Sequence stamp: \`$STAMP\`"
    echo "- Queue log: \`$QUEUE_LOG\`"
    echo "- GPU free-memory threshold: \`${MIN_FREE_MB} MiB\`"
    echo "- Order: B3 hard_500 seed 42 -> multi-seed baseline -> ablation groups"
    echo
    echo "## Planned Stages"
    echo
    echo "1. B3: \`hotpot_hard_500\`, seed \`42\`, same-payload V3/V4/V5/V6."
    echo "2. Multi-seed: \`hotpot_all_1000\`, \`hotpot_hard_1000\`, \`hotpot_hard_500\` with seeds \`43\` and \`44\`."
    echo "3. Ablation groups: \`pooler\`, \`layerwise\`, \`score_mode\`, \`hard_weight\`, \`adaptive\`."
    echo
    echo "## Expected Output Families"
    echo
    echo "- B3 raw CSV: \`experiments/v6_hp_hyper_next/results/same_payload_b3_hard500_raw.csv\`"
    echo "- Multi-seed raw CSV: \`experiments/v6_hp_hyper_next/results/same_payload_multiseed_raw.csv\`"
    echo "- Ablation raw/summary/report files under \`experiments/v6_hp_hyper_next/results/\` and \`experiments/v6_hp_hyper_next/reports/\`."
    echo
    echo "## Notes"
    echo
    echo "The script waits for a safe GPU window before each stage and does not preempt other users' GPU jobs."
  } > "$report"
  log_msg "[report-written] $report"
}

log_msg "[sequence-start] $(date '+%F %T %Z') stamp=$STAMP"
write_sequence_report

run_same_payload_stage \
  "B3_hard500_seed42" \
  "hotpot_hard_500" \
  "42" \
  "$ROOT_DIR/experiments/v6_hp_hyper_next/results/same_payload_b3_hard500_raw.csv" \
  "v6_hp_hyper_next_b3" \
  "same_payload_b3"

for seed in 43 44; do
  for subset in hotpot_all_1000 hotpot_hard_1000 hotpot_hard_500; do
    run_same_payload_stage \
      "multiseed_${subset}_seed${seed}" \
      "$subset" \
      "$seed" \
      "$ROOT_DIR/experiments/v6_hp_hyper_next/results/same_payload_multiseed_raw.csv" \
      "v6_hp_hyper_next_multiseed" \
      "same_payload_multiseed"
  done
done

for group in pooler layerwise score_mode hard_weight adaptive; do
  run_ablation_stage "$group"
done

log_msg "[sequence-done] $(date '+%F %T %Z') stamp=$STAMP"
