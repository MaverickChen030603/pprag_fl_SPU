#!/usr/bin/env bash
# U1 is retrieval-only: frozen route, frozen local rankers, no reader.
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 <2wikimultihopqa|musique> <cuda-device>" >&2; exit 2; fi
DATASET="$1"; DEVICE="$2"
case "$DATASET" in 2wikimultihopqa|musique) ;; *) exit 2;; esac
ROOT=/home/iiserver31/projects/FedE4RAG-main
PY=/home/iiserver31/anaconda3/envs/supv2/bin/python
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
PREV="$V20/multidataset_depth_calibration"
OUT="$V20/stage_u1_factorized_audit/$DATASET"
SPLIT="$PREV/frozen_splits/${DATASET}_development_101_400.jsonl"
ROUTE="$PREV/inherited_routes/${DATASET}_topic_silo_bc3.jsonl"
INDEX="$V17/retrieval/local_indexes/$DATASET/topic_silo"
ASSIGN="$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl"
POOL="$OUT/local_retrievers/all_client_rankers_depth10.jsonl"
mkdir -p "$OUT/local_retrievers"
if [[ ! -f "$POOL.manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$V20/stage_u1_factorized_audit/build_local_ranker_pool.py" --dataset "$DATASET" --split "$SPLIT" --local-index-root "$INDEX" --output "$POOL" --device cuda --resume
fi
for RUN in run1 run2; do
  if [[ ! -f "$OUT/$RUN/reports/next_method_decision.json" ]]; then
    "$PY" "$V20/stage_u1_factorized_audit/run_factorized_audit.py" --dataset "$DATASET" --split "$SPLIT" --pool "$POOL" --actual-route "$ROUTE" --assignment "$ASSIGN" --output-root "$OUT/$RUN"
  fi
done
cmp -s "$OUT/run1/factorial_matrix/routing_local_matrix.csv" "$OUT/run2/factorial_matrix/routing_local_matrix.csv"
cmp -s "$OUT/run1/factorial_matrix/per_query_results.jsonl" "$OUT/run2/factorial_matrix/per_query_results.jsonl"
printf '{"dataset":"%s","byte_identical_matrix":true,"byte_identical_per_query":true,"reader_started":false}\n' "$DATASET" > "$OUT/reproducibility.json"
