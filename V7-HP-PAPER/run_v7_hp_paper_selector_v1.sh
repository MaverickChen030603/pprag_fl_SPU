#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:."

PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" V7-HP-PAPER/run_support_insertion_selector_v1.py \
  --validation V7-HP4/data/hotpot_validation_1000.json \
  --policy-a V7-HP4/outputs/hp4_phase3_100_batch/policy_a/hp4_policy_reinforce.pt \
  --output-root V7-HP-PAPER/outputs \
  --report-dir V7-HP-PAPER/reports \
  --reader-model "${READER_MODEL:-google/flan-t5-large}" \
  --device "${DEVICE:-cuda:0}" \
  --reader-batch-size "${READER_BATCH_SIZE:-1}" \
  --sample-size "${SAMPLE_SIZE:-100}" \
  --predictor-model "${PREDICTOR_MODEL:-rf}" \
  --threshold "${SELECTOR_THRESHOLD:-0.70}" \
  --risk-threshold "${RISK_THRESHOLD:-0.38}"
