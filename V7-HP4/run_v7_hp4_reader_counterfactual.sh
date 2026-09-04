#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
READER_MODEL="${READER_MODEL:-google/flan-t5-large}"
DEVICE="${DEVICE:-cuda:2}"
READER_BSZ="${READER_BSZ:-2}"
TOP_K="${TOP_K:-5}"
ALPHA="${ALPHA:-0.55}"
MAX_DEV="${MAX_DEV:-300}"

mkdir -p V7-HP4/logs V7-HP4/outputs/hp4_reader_counterfactual "实验分析报告/V7-HP4"

PYTHONPATH=. "$PYTHON_BIN" V7-HP4/run_hp4_reader_counterfactual_eval.py \
  --micro data/v7_hp4_micro_benchmark.json \
  --dev300 V7-HP3/data/hotpot_dev_stratified_300.json \
  --output-root V7-HP4/outputs/hp4_reader_counterfactual \
  --report-dir "实验分析报告/V7-HP4" \
  --reader-model "$READER_MODEL" \
  --device "$DEVICE" \
  --reader-batch-size "$READER_BSZ" \
  --top-k "$TOP_K" \
  --alpha "$ALPHA" \
  --max-dev "$MAX_DEV"
