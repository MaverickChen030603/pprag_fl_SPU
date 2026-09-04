#!/usr/bin/env bash
set -euo pipefail

ROOT="${V15_ROOT:-/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v15_robust_context_repair}"
PROJECT="${FEDE4RAG_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
HOTPOT_GLOB="${HOTPOT_GLOB:-/home/iiserver31/.cache/huggingface/datasets/hotpot_qa/distractor/**/*.arrow}"
TWOWIKI_ZIP="${TWOWIKI_ZIP:-${PROJECT}/V7-HP-PAPER/cross_dataset_validation/raw/2wiki/data.zip}"

cd "$ROOT"
mkdir -p logs retrieval/indexes

"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" protocol/01_protocol_and_data_audit.py \
  --project-root "$PROJECT" --max-file-mb 32
"$PYTHON" protocol/00_prepare_fresh_sources.py \
  --hotpot-arrow-glob "$HOTPOT_GLOB" \
  --two-wiki-archive "$TWOWIKI_ZIP"
"$PYTHON" protocol/02_freeze_data_splits.py \
  --hotpot-source data/sources/hotpotqa_distractor_train.jsonl \
  --two-wiki-source data/sources/2wikimultihopqa_train.json
"$PYTHON" retrieval/01_build_corpus_index.py \
  --source data/sources/hotpotqa_distractor_train.jsonl \
  --dataset hotpotqa --output retrieval/indexes/hotpotqa.sqlite
"$PYTHON" retrieval/01_build_corpus_index.py \
  --source data/sources/2wikimultihopqa_train.json \
  --dataset 2wikimultihopqa --output retrieval/indexes/2wikimultihopqa.sqlite

echo "V15 phase 0 complete"

