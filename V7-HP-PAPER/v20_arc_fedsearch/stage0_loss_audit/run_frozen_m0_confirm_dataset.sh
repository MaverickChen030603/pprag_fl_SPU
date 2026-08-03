#!/usr/bin/env bash
# Frozen V20 Stage M0-Confirm replay for one development-only dataset.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <2wikimultihopqa|musique> <cuda-device>" >&2
  exit 2
fi

DATASET="$1"
DEVICE="$2"
case "$DATASET" in
  2wikimultihopqa|musique) ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

ROOT="/home/iiserver31/projects/FedE4RAG-main"
PY="/home/iiserver31/anaconda3/envs/supv2/bin/python"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
OUT="$V20/multidataset_depth_calibration/${DATASET}_n300"
FROZEN="$V20/multidataset_depth_calibration/frozen_splits/${DATASET}_development_101_400.jsonl"
ROUTES="$V20/multidataset_depth_calibration/inherited_routes/${DATASET}_topic_silo_bc3.jsonl"
DEPTH="$OUT/all_client_local_depth10.jsonl"
SOURCE="$V17/data/$DATASET/development.jsonl"
ASSIGNMENT="$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl"
CENTROIDS="$V17/partitions/centroids/$DATASET/topic_silo_m20.npy"
LOCAL_INDEX="$V17/retrieval/local_indexes/$DATASET/topic_silo"
ORIGINS="$V17/partitions/client_query_distribution.csv"

mkdir -p "$OUT" "$(dirname "$FROZEN")" "$(dirname "$ROUTES")"

if [[ ! -f "$FROZEN" ]]; then
  "$PY" "$V20/stage0_loss_audit/build_frozen_eval_slice.py" \
    --dataset "$DATASET" --source "$SOURCE" --output "$FROZEN" --start-index 100 --count 300
fi
if [[ ! -f "$ROUTES" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$V20/stage0_loss_audit/build_frozen_topic_routes.py" \
    --dataset "$DATASET" --split "$FROZEN" --origins "$ORIGINS" --centroids "$CENTROIDS" --output "$ROUTES" --device cuda
fi
if [[ ! -f "$DEPTH.manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$V20/stage0_loss_audit/build_all_client_depth_pool.py" \
    --dataset "$DATASET" --split "$FROZEN" --inherited-pool "$ROUTES" --local-index-root "$LOCAL_INDEX" \
    --output "$DEPTH" --local-depth 10 --sparse-candidates 100 --max-queries 300 --device cuda --resume
fi
for RUN in run1 run2; do
  if [[ ! -f "$OUT/$RUN/m0_m1_go_no_go.json" ]]; then
    "$PY" "$V20/stage0_loss_audit/run_m0_m1_matrix.py" \
      --dataset "$DATASET" --split "$FROZEN" --depth-pool "$DEPTH" --inherited-pool "$ROUTES" \
      --assignment "$ASSIGNMENT" --output-dir "$OUT/$RUN"
  fi
done
cmp -s "$OUT/run1/allocation_merge_matrix.csv" "$OUT/run2/allocation_merge_matrix.csv"
cmp -s "$OUT/run1/per_query_allocation_merge.csv" "$OUT/run2/per_query_allocation_merge.csv"
printf '{\n  "dataset": "%s",\n  "byte_identical_matrix": true,\n  "byte_identical_per_query": true,\n  "reader_started": false\n}\n' "$DATASET" > "$OUT/reproducibility.json"
echo "V20 M0-Confirm complete: $DATASET"
