#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
GPU="${GPU:-1}"
READER_GPU="${READER_GPU:-2}"
ROUNDS="${ROUNDS:-12}"
SEEDS="${SEEDS:-0,1,2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_RAW="${TRAIN_RAW:-/home/iiserver31/projects/FedE4RAG-main/FedE/select_data_hotpot_train_5000.json}"
DEV_SAMPLE="${DEV_SAMPLE:-V7-HP3/data/hotpot_dev_stratified_300.json}"
MAX_EVAL="${MAX_EVAL:-300}"

"$PYTHON_BIN" V7-HP3/run_experiment_suite.py \
  --suite hp2_reader_aligned \
  --rounds "$ROUNDS" \
  --seed-list "$SEEDS" \
  --gpu "$GPU" \
  --batch-size "$BATCH_SIZE" \
  --experiment-name pprag_fl_v7_hp3 \
  --rawdata-path "$TRAIN_RAW" \
  --rag-dataset hotpot_qa \
  --rag-hotpot-split validation \
  --rag-hotpot-max-examples "$MAX_EVAL"

"$PYTHON_BIN" V7-HP3/run_all_hotpot_official_eval.py \
  --upstream-root V7-HP3/outputs/pprag_fl_v7_hp3 \
  --output-root V7-HP3/outputs/hotpot_official_fullwiki_dev300 \
  --rawdata-path "$DEV_SAMPLE" \
  --suite hp2_reader_aligned \
  --max-examples "$MAX_EVAL" \
  --support-topk 2 \
  --answer-topk 5 \
  --batch-size 128 \
  --device "cuda:$GPU" \
  --prefer-official

"$PYTHON_BIN" V7-HP3/run_all_hotpot_reader_eval.py \
  --upstream-root V7-HP3/outputs/pprag_fl_v7_hp3 \
  --output-root V7-HP3/outputs/hotpot_reader_fullwiki_t5small_dev300 \
  --rawdata-path "$DEV_SAMPLE" \
  --suite hp2_reader_aligned \
  --max-examples "$MAX_EVAL" \
  --prefer-official \
  --reader-model google-t5/t5-small \
  --local-reader-only \
  --retrieval-topk 5 \
  --support-topk 2 \
  --batch-size 128 \
  --reader-batch-size 16 \
  --device "cuda:$READER_GPU"

"$PYTHON_BIN" V7-HP3/write_hp2_reader_alignment_analysis.py
