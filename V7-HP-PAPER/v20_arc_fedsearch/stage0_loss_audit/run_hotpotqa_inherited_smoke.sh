#!/usr/bin/env bash
set -euo pipefail

# Retrieval-only audit.  It replays frozen V17 pools; it does not invoke a reader.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V17_ROOT="${V17_ROOT:-$ROOT/../v17_fedaction_rag}"
OUT="$ROOT/stage0_loss_audit/hotpotqa_inherited_n100"
# The server's experiment dependencies live in supv2; callers may still
# override this for a portable local environment.
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"

mkdir -p "$OUT"
"$PYTHON" "$ROOT/stage0_loss_audit/run_inherited_replay.py" \
  --dataset hotpotqa \
  --split "$V17_ROOT/data/hotpotqa/development.jsonl" \
  --federated-pool "$V17_ROOT/oracle/phase_a_checkpoint100/pools/hotpotqa_topic_silo.jsonl" \
  --centralized-pool "$V17_ROOT/oracle/phase_a_checkpoint100/pools/hotpotqa_centralized.jsonl" \
  --assignment "$V17_ROOT/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" \
  --output-dir "$OUT" \
  --max-queries 100
