#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
SUITES="${SUITES:-hp1_budget_aligned}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
DEVICE="${DEVICE:-cuda:1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-V7-HP1/outputs/hotpot_official_eval}"
RAW_DATA="${RAW_DATA:-FedE/select_data_hotpot_train_5000.json}"
PREFER_OFFICIAL="${PREFER_OFFICIAL:-0}"
FORCE="${FORCE:-0}"

args=(
  --upstream-root V7-HP1/outputs/pprag_fl_v7_hp1
  --output-root "$OUTPUT_ROOT"
  --rawdata-path "$RAW_DATA"
  --suite "$SUITES"
  --max-examples "$MAX_EXAMPLES"
  --support-topk 2
  --answer-topk 5
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ "$PREFER_OFFICIAL" == "1" ]]; then
  args+=(--prefer-official)
fi
if [[ "$FORCE" == "1" ]]; then
  args+=(--force)
fi

"$PYTHON_BIN" V7-HP1/run_all_hotpot_official_eval.py "${args[@]}"
"$PYTHON_BIN" V7-HP1/write_hotpot_official_analysis.py --output-root "$OUTPUT_ROOT" --report-dir 实验分析报告/V7-HP1
