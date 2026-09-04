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
HARD_SAMPLE="${HARD_SAMPLE:-V7-HP3/data/hotpot_recoverable_hard100.json}"
MAX_EVAL="${MAX_EVAL:-100}"
READER_MODEL="${READER_MODEL:-google/flan-t5-large}"
FALLBACK_READER_MODEL="${FALLBACK_READER_MODEL:-google/flan-t5-large}"
READER_BSZ="${READER_BSZ:-2}"

"$PYTHON_BIN" V7-HP3/prepare_recoverable_hard100.py --target-size "$MAX_EVAL" --output "$HARD_SAMPLE" --meta-output V7-HP3/data/hotpot_recoverable_hard100.meta.json

if [[ "${FORCE_REBUILD_TASK:-1}" == "1" ]]; then
  rm -rf FedE/num5_dir_a03_imb00_ts0_v7hp3
fi

"$PYTHON_BIN" V7-HP3/run_experiment_suite.py \
  --suite hp3_reset_hard \
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
  --output-root V7-HP3/outputs/hotpot_official_fullwiki_hard100 \
  --rawdata-path "$HARD_SAMPLE" \
  --suite hp3_reset_hard \
  --max-examples "$MAX_EVAL" \
  --support-topk 2 \
  --answer-topk 5 \
  --batch-size 128 \
  --device "cuda:$GPU" \
  --prefer-official

"$PYTHON_BIN" V7-HP3/run_all_hotpot_reader_eval.py \
  --upstream-root V7-HP3/outputs/pprag_fl_v7_hp3 \
  --output-root V7-HP3/outputs/hotpot_reader_strong_hard100 \
  --rawdata-path "$HARD_SAMPLE" \
  --suite hp3_reset_hard \
  --max-examples "$MAX_EVAL" \
  --prefer-official \
  --reader-model "$READER_MODEL" \
  --fallback-reader-model "$FALLBACK_READER_MODEL" \
  --retrieval-topk 5 \
  --support-topk 2 \
  --batch-size 128 \
  --reader-batch-size "$READER_BSZ" \
  --device "cuda:$READER_GPU"

"$PYTHON_BIN" V7-HP3/write_hp3_reset_analysis.py
