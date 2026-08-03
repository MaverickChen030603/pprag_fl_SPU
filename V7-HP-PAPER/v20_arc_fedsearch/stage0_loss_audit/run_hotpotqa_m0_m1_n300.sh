#!/usr/bin/env bash
set -euo pipefail

# Frozen M0/M1 confirmation.  Retrieval only: no reader invocation is allowed.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V17_ROOT="${V17_ROOT:-$ROOT/../v17_fedaction_rag}"
V19_N300="${V19_N300:-$ROOT/../v19_reader_aligned_selective_update/stage0b_top5_boundary_crossing/retrieval_confirmation}"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
DEPTH_DIR="$ROOT/stage0_loss_audit/hotpotqa_depth10_n300"
DEPTH_POOL="$DEPTH_DIR/all_client_local_depth10.jsonl"
MATRIX_DIR="$ROOT/stage0_loss_audit/hotpotqa_m0_m1_n300"
mkdir -p "$DEPTH_DIR" "$MATRIX_DIR"

"$PYTHON" "$ROOT/stage0_loss_audit/build_all_client_depth_pool.py" \
  --dataset hotpotqa \
  --split "$V19_N300/hotpotqa_development_disjoint_101_400.jsonl" \
  --inherited-pool "$V19_N300/pools/hotpotqa_topic_silo_n300.jsonl" \
  --local-index-root "$V17_ROOT/retrieval/local_indexes/hotpotqa/topic_silo" \
  --output "$DEPTH_POOL" \
  --local-depth 10 \
  --resume

for run in run1 run2; do
  "$PYTHON" "$ROOT/stage0_loss_audit/run_m0_m1_matrix.py" \
    --dataset hotpotqa \
    --split "$V19_N300/hotpotqa_development_disjoint_101_400.jsonl" \
    --depth-pool "$DEPTH_POOL" \
    --inherited-pool "$V19_N300/pools/hotpotqa_topic_silo_n300.jsonl" \
    --assignment "$V17_ROOT/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" \
    --output-dir "$MATRIX_DIR/$run"
done

cmp "$MATRIX_DIR/run1/allocation_merge_matrix.csv" "$MATRIX_DIR/run2/allocation_merge_matrix.csv"
cmp "$MATRIX_DIR/run1/per_query_allocation_merge.csv" "$MATRIX_DIR/run2/per_query_allocation_merge.csv"
printf '%s\n' '{"status":"deterministic_repeat_pass","reader_started":false}' > "$MATRIX_DIR/repeatability.json"
