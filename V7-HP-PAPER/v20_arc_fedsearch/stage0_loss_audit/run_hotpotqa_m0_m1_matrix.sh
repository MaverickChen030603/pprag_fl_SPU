#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V17_ROOT="${V17_ROOT:-$ROOT/../v17_fedaction_rag}"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
DEPTH="$ROOT/stage0_loss_audit/hotpotqa_depth10_n100/all_client_local_depth10.jsonl"
BASE="$ROOT/stage0_loss_audit/hotpotqa_m0_m1_n100"

for run in run1 run2; do
  "$PYTHON" "$ROOT/stage0_loss_audit/run_m0_m1_matrix.py" \
    --dataset hotpotqa \
    --split "$V17_ROOT/data/hotpotqa/development.jsonl" \
    --depth-pool "$DEPTH" \
    --inherited-pool "$V17_ROOT/oracle/phase_a_checkpoint100/pools/hotpotqa_topic_silo.jsonl" \
    --assignment "$V17_ROOT/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" \
    --output-dir "$BASE/$run"
done

cmp "$BASE/run1/allocation_merge_matrix.csv" "$BASE/run2/allocation_merge_matrix.csv"
cmp "$BASE/run1/per_query_allocation_merge.csv" "$BASE/run2/per_query_allocation_merge.csv"
printf '%s\n' '{"status":"deterministic_repeat_pass","reader_started":false}' > "$BASE/repeatability.json"
