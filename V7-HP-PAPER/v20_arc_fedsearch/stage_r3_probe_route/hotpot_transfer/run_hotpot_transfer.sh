#!/usr/bin/env bash
# Run the frozen R3-T/R3-C Hotpot transfer. Reader and final test are forbidden.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cuda-device>" >&2
  exit 2
fi

DEVICE="$1"
ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
STAGE="$V20/stage_r3_probe_route"
TRANSFER="$STAGE/hotpot_transfer"
PROTO="$TRANSFER/protocol"
PACKETS="$TRANSFER/packets"
MODELS="$TRANSFER/models"
HOLDOUT="$TRANSFER/holdout"
PROFILES="$TRANSFER/resource_profiles"
ROUTES="$HOLDOUT/inherited_b3_routes.jsonl"

mkdir -p "$PROTO" "$PACKETS" "$MODELS" "$HOLDOUT" "$PROFILES"
if [[ ! -f "$PROTO/transfer_split_manifest.json" ]]; then
  "$PY" "$TRANSFER/prepare_hotpot_transfer_splits.py" \
    --train "$V17/data/hotpotqa/train.jsonl" \
    --development "$V17/data/hotpotqa/development.jsonl" \
    --prior-used "$V20/stage0_loss_audit/hotpotqa_depth10_n100/all_client_local_depth10.jsonl" \
    --prior-used "$V20/stage0_loss_audit/hotpotqa_depth10_n300/all_client_local_depth10.jsonl" \
    --prior-used "$ROOT/V7-HP-PAPER/v19_reader_aligned_selective_update/stage0b_top5_boundary_crossing/retrieval_confirmation/hotpotqa_development_disjoint_101_400.jsonl" \
    --output-dir "$PROTO"
fi
if [[ ! -f "$PROFILES/client_profiles.json" ]]; then
  "$PY" "$TRANSFER/build_hotpot_p0_profiles.py" \
    --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" \
    --p0-centroids "$V17/partitions/centroids/hotpotqa/topic_silo_m20.npy" --output-dir "$PROFILES"
fi
if [[ ! -f "$PACKETS/probe_train.manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/materialize_candidate_probe_packets.py" \
    --dataset hotpotqa --split "$PROTO/probe_train.jsonl" --profiles "$PROFILES/client_profiles.json" \
    --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" --output "$PACKETS/probe_train.jsonl" --device cuda --resume
fi
if [[ ! -f "$MODELS/model_results.csv" ]]; then
  "$PY" "$STAGE/train_evaluate_logistic_ranker.py" --mode train --dataset hotpotqa --split "$PROTO/probe_train.jsonl" \
    --packets "$PACKETS/probe_train.jsonl" --assignment "$V17/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" --output-dir "$MODELS"
fi
if [[ ! -f "$ROUTES" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$V20/stage0_loss_audit/build_frozen_topic_routes.py" \
    --dataset hotpotqa --split "$PROTO/probe_holdout.jsonl" --origins "$V17/partitions/client_query_distribution.csv" \
    --centroids "$V17/partitions/centroids/hotpotqa/topic_silo_m20.npy" --output "$ROUTES" --client-budget 3 --device cuda
fi
if [[ ! -f "$PACKETS/probe_holdout.manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/materialize_candidate_probe_packets.py" \
    --dataset hotpotqa --split "$PROTO/probe_holdout.jsonl" --profiles "$PROFILES/client_profiles.json" \
    --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" --inherited-routes "$ROUTES" \
    --output "$PACKETS/probe_holdout.jsonl" --device cuda --resume
fi
"$PY" "$STAGE/train_evaluate_logistic_ranker.py" --mode evaluate --dataset hotpotqa --split "$PROTO/probe_holdout.jsonl" \
  --packets "$PACKETS/probe_holdout.jsonl" --assignment "$V17/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" \
  --models-dir "$MODELS" --output-dir "$HOLDOUT/main_results"
PYTHONPATH="$STAGE" "$PY" "$TRANSFER/evaluate_cost_matched_baselines.py" --split "$PROTO/probe_holdout.jsonl" \
  --packets "$PACKETS/probe_holdout.jsonl" --assignment "$V17/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" --output-dir "$HOLDOUT/cost_matched"
