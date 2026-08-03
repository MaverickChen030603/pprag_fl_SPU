#!/usr/bin/env bash
set -euo pipefail

# This is a retrieval-only local-depth audit.  It never invokes a reader.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V17_ROOT="${V17_ROOT:-$ROOT/../v17_fedaction_rag}"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
OUT="$ROOT/stage0_loss_audit/hotpotqa_depth10_n100"
POOL="$OUT/all_client_local_depth10.jsonl"
mkdir -p "$OUT"

"$PYTHON" "$ROOT/stage0_loss_audit/build_all_client_depth_pool.py" \
  --dataset hotpotqa \
  --split "$V17_ROOT/data/hotpotqa/development.jsonl" \
  --inherited-pool "$V17_ROOT/oracle/phase_a_checkpoint100/pools/hotpotqa_topic_silo.jsonl" \
  --local-index-root "$V17_ROOT/retrieval/local_indexes/hotpotqa/topic_silo" \
  --output "$POOL" \
  --local-depth 10 \
  --max-queries 100 \
  --resume

"$PYTHON" "$ROOT/stage0_loss_audit/run_inherited_replay.py" \
  --dataset hotpotqa \
  --split "$V17_ROOT/data/hotpotqa/development.jsonl" \
  --federated-pool "$POOL" \
  --centralized-pool "$V17_ROOT/oracle/phase_a_checkpoint100/pools/hotpotqa_centralized.jsonl" \
  --assignment "$V17_ROOT/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" \
  --output-dir "$OUT/replay" \
  --max-queries 100 \
  --actual-local-k 5
