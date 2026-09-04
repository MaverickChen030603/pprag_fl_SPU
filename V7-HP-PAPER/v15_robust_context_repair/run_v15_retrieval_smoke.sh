#!/usr/bin/env bash
set -euo pipefail

ROOT="${V15_ROOT:-/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v15_robust_context_repair}"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
BGE="${BGE:-/home/iiserver31/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a}"
CE="${CE:-/home/iiserver31/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/c5ee24cb16019beea0893ab7796b1df96625c6b8}"
N="${SMOKE_QUERIES:-50}"

export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"

for spec in "hotpotqa:0" "2wikimultihopqa:1"; do
  dataset="${spec%%:*}"
  gpu="${spec##*:}"
  out="retrieval/smoke_${dataset}_${N}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" retrieval/02_generate_true_pools.py \
    --split "data/${dataset}/development.jsonl" \
    --dataset "$dataset" --index "retrieval/indexes/${dataset}.sqlite" \
    --output-dir "$out" --encoder "$BGE" --max-queries "$N"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" retrieval/03_score_crossencoder.py \
    --split "data/${dataset}/development.jsonl" --pool "$out/top10.jsonl" \
    --output "$out/top10_ce.jsonl" --checkpoint "$CE"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" retrieval/03_score_crossencoder.py \
    --split "data/${dataset}/development.jsonl" --pool "$out/top20.jsonl" \
    --output "$out/top20_ce.jsonl" --checkpoint "$CE"
  "$PYTHON" action_generation/01_generate_action_pools.py \
    --pool "$out/top10_ce.jsonl" --output "action_generation/smoke_${dataset}_${N}/actions_top10.jsonl"
  "$PYTHON" action_generation/01_generate_action_pools.py \
    --pool "$out/top20_ce.jsonl" --output "action_generation/smoke_${dataset}_${N}/actions_top20.jsonl"
done

"$PYTHON" retrieval/04_summarize_pools.py \
  --statistics retrieval/smoke_hotpotqa_${N}/retrieval_pool_statistics.csv \
               retrieval/smoke_2wikimultihopqa_${N}/retrieval_pool_statistics.csv \
  --output reports/retrieval_pool_recall.md

