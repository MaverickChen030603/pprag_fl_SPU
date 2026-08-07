#!/usr/bin/env bash
# Execute the frozen, reader-free R3 lightweight logistic-ranker protocol.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <2wikimultihopqa|musique> <cuda-device>" >&2
  exit 2
fi

DATASET="$1"
DEVICE="$2"
ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
STAGE="$V20/stage_r3_probe_route"
R2="$V20/stage_r2_mars_route/$DATASET"
R2A6="$V20/stage_r2a6_resource_memory"
PROTO="$STAGE/protocol/$DATASET"
PACKETS="$STAGE/ranker_training/packets/$DATASET"
MODELS="$STAGE/ranker_training/models/$DATASET"
HOLDOUT="$STAGE/holdout/$DATASET/main_results"
ROUTES="$STAGE/holdout/$DATASET/inherited_b3_routes.jsonl"

mkdir -p "$PROTO" "$PACKETS" "$MODELS" "$HOLDOUT" "$(dirname "$ROUTES")"
"$PY" "$STAGE/protocol/prepare_ranker_splits.py" --dataset "$DATASET" --r2-root "$R2" --r2a6-root "$R2A6" --output-dir "$PROTO"

if [[ ! -f "$PACKETS/probe_train.manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/materialize_candidate_probe_packets.py" \
    --dataset "$DATASET" --split "$PROTO/probe_train.jsonl" --profiles "$R2/resource_profiles/client_profiles.json" \
    --local-index-root "$V17/retrieval/local_indexes/$DATASET/topic_silo" --output "$PACKETS/probe_train.jsonl" --device cuda --resume
fi
if [[ ! -f "$MODELS/model_results.csv" ]]; then
  "$PY" "$STAGE/train_evaluate_logistic_ranker.py" --mode train --dataset "$DATASET" --split "$PROTO/probe_train.jsonl" \
    --packets "$PACKETS/probe_train.jsonl" --assignment "$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl" --output-dir "$MODELS"
fi
if [[ ! -f "$ROUTES" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$V20/stage0_loss_audit/build_frozen_topic_routes.py" \
    --dataset "$DATASET" --split "$PROTO/probe_holdout.jsonl" --origins "$V17/partitions/client_query_distribution.csv" \
    --centroids "$V17/partitions/centroids/$DATASET/topic_silo_m20.npy" --output "$ROUTES" --client-budget 3 --device cuda
fi
if [[ ! -f "$PACKETS/probe_holdout.manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/materialize_candidate_probe_packets.py" \
    --dataset "$DATASET" --split "$PROTO/probe_holdout.jsonl" --profiles "$R2/resource_profiles/client_profiles.json" \
    --local-index-root "$V17/retrieval/local_indexes/$DATASET/topic_silo" --inherited-routes "$ROUTES" \
    --output "$PACKETS/probe_holdout.jsonl" --device cuda --resume
fi
"$PY" "$STAGE/train_evaluate_logistic_ranker.py" --mode evaluate --dataset "$DATASET" --split "$PROTO/probe_holdout.jsonl" \
  --packets "$PACKETS/probe_holdout.jsonl" --assignment "$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl" \
  --models-dir "$MODELS" --output-dir "$HOLDOUT"
