#!/usr/bin/env bash
set -euo pipefail

# V19 Stage 0 smoke: validates the adapter->frozen routed pool contract.
# This deliberately uses development only and writes reader-compatible contexts.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$(cd "$ROOT/../.." && pwd)"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
DATASET="${DATASET:-hotpotqa}"
SEED="${SEED:-20260731}"
OUT="$ROOT/stage0_full_upload/smoke_${DATASET}_seed${SEED}"
POOL="$PROJECT/v17_fedaction_rag/oracle/phase_a_checkpoint100/pools/${DATASET}_topic_silo.jsonl"
ASSIGNMENT="$PROJECT/v17_fedaction_rag/partitions/assignments/${DATASET}/topic_silo_m20.jsonl"
DATA="$PROJECT/v17_fedaction_rag/data/${DATASET}"

for METHOD in frozen centralized fedavg fedprox scaffold; do
  "$PYTHON" "$ROOT/stage0_full_upload/run_stage0_viability.py" \
    --dataset "$DATASET" --method "$METHOD" --train "$DATA/train.jsonl" \
    --development "$DATA/development.jsonl" --development-pool "$POOL" --assignment "$ASSIGNMENT" \
    --output-dir "$OUT/$METHOD" --seed "$SEED"
done
