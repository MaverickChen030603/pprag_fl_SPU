#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
READER_MODEL="${READER_MODEL:-google/flan-t5-large}"
DEVICE="${DEVICE:-cuda:2}"
MAX_DEV="${MAX_DEV:-1000}"
TOP_K="${TOP_K:-5}"
ALPHA="${ALPHA:-0.55}"
READER_BSZ="${READER_BSZ:-2}"
PERMUTATION_ROUNDS="${PERMUTATION_ROUNDS:-10000}"

mkdir -p V7-HP4/logs V7-HP4/outputs/hp4_full_validation V7-HP4/outputs/hp4_routing_visualization "实验分析报告/V7-HP4"

echo "[Task2] Full 1000+ reader validation + paired significance"
PYTHONPATH=. "$PYTHON_BIN" V7-HP4/run_hp4_full_validation_eval.py \
  --preferred-dev V7-HP4/data/hotpot_validation_1000.json \
  --generated-dev V7-HP4/data/hotpot_validation_1000.json \
  --fallback-dev FedE/select_data_hotpot_train_5000.json \
  --output-root V7-HP4/outputs/hp4_full_validation \
  --report-dir "实验分析报告/V7-HP4" \
  --reader-model "$READER_MODEL" \
  --device "$DEVICE" \
  --reader-batch-size "$READER_BSZ" \
  --top-k "$TOP_K" \
  --alpha "$ALPHA" \
  --max-dev "$MAX_DEV" \
  --permutation-rounds "$PERMUTATION_ROUNDS"

echo "[Task3] Routing dataframe + case studies"
PYTHONPATH=. "$PYTHON_BIN" V7-HP4/export_hp4_routing_case_studies.py \
  --dev V7-HP4/data/hotpot_validation_1000.json \
  --eval-rows V7-HP4/outputs/hp4_full_validation/full_validation_reader_rows.json \
  --output-root V7-HP4/outputs/hp4_routing_visualization \
  --report-dir "实验分析报告/V7-HP4" \
  --top-k "$TOP_K" \
  --alpha "$ALPHA" \
  --max-dev "$MAX_DEV"
