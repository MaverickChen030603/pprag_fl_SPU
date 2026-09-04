#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
TOP_K="${TOP_K:-5}"
ALPHA="${ALPHA:-0.55}"
MAX_DEV="${MAX_DEV:-300}"

mkdir -p V7-HP4/logs V7-HP4/outputs/hp4_full "实验分析报告/V7-HP4"

echo "[1/4] prepare HP4 micro-benchmark"
"$PYTHON_BIN" V7-HP4/prepare_hp4_micro_benchmark.py \
  --input V7-HP3/data/hotpot_dev_stratified_300.json \
  --output data/v7_hp4_micro_benchmark.json \
  --target-size 30
cp -f data/v7_hp4_micro_benchmark.json V7-HP4/data/v7_hp4_micro_benchmark.json
cp -f data/v7_hp4_micro_benchmark.meta.json V7-HP4/data/v7_hp4_micro_benchmark.meta.json

echo "[2/4] Top-K Context Delta Audit"
PYTHONPATH=src "$PYTHON_BIN" tests/test_context_delta.py

echo "[3/4] run HP4 soft-hybrid full proxy evaluation"
PYTHONPATH=. "$PYTHON_BIN" V7-HP4/run_hp4_full_experiment.py \
  --micro data/v7_hp4_micro_benchmark.json \
  --dev300 V7-HP3/data/hotpot_dev_stratified_300.json \
  --output-root V7-HP4/outputs/hp4_full \
  --report-dir "实验分析报告/V7-HP4" \
  --top-k "$TOP_K" \
  --alpha "$ALPHA" \
  --max-dev "$MAX_DEV"

echo "[4/4] done"
