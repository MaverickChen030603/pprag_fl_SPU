#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 <2wikimultihopqa|musique> <cuda-device>" >&2; exit 2; fi
DATASET="$1"; DEVICE="$2"
ROOT=/home/iiserver31/projects/FedE4RAG-main
PY=/home/iiserver31/anaconda3/envs/supv2/bin/python
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
R2="$V20/stage_r2_mars_route/$DATASET"
OUT="$V20/stage_r2a5_candidate_compression/$DATASET"
SPLIT="$R2/protocol/router_dev_smoke100.jsonl"
POOL="$OUT/local_pool/all_client_rankers_depth10.jsonl"
mkdir -p "$OUT/local_pool"
if [[ ! -f "$POOL.manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$V20/stage_u1_factorized_audit/build_local_ranker_pool.py" --dataset "$DATASET" --split "$SPLIT" --local-index-root "$V17/retrieval/local_indexes/$DATASET/topic_silo" --output "$POOL" --device cuda --resume
fi
BEST_P=8
if [[ "$DATASET" == "musique" ]]; then BEST_P=16; fi
CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$V20/stage_r2a5_candidate_compression/run_compression_audit.py" --dataset "$DATASET" --split "$SPLIT" --profiles "$R2/resource_profiles/client_profiles.json" --assignment "$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl" --local-pool "$POOL" --output-root "$OUT" --best-p "$BEST_P"
