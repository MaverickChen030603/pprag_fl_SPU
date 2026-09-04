#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$ROOT/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
WARMUP="${WARMUP:-50}"
SAMPLES="${SAMPLES:-500}"

mkdir -p "$ROOT/logs" "$ROOT/outputs/cost"
cd "$PROJECT"

systems=(
  frozen_top5_baseline
  full_v4
  lite_lexical_pair
  baseline_truncated_660
  recomp_top1
  recomp_budgetmatched
)

for system in "${systems[@]}"; do
  "$PYTHON_BIN" -u "$ROOT/measure_frozen_inference_cost.py" \
    --system "$system" \
    --device "$DEVICE" \
    --warmup "$WARMUP" \
    --samples "$SAMPLES" \
    2>&1 | tee "$ROOT/logs/frozen_cost_${system}.log"
done

"$PYTHON_BIN" -u "$ROOT/measure_frozen_inference_cost.py" --combine \
  2>&1 | tee "$ROOT/logs/frozen_cost_combine.log"

printf 'complete\n' > "$ROOT/logs/frozen_cost_benchmark.done"
