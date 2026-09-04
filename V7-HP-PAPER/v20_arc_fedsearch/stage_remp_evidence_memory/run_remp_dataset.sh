#!/usr/bin/env bash
# REM-P recovery gate: profile representation and candidate recall only.
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

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
STAGE="$V20/stage_remp_evidence_memory"
R2="$V20/stage_r2_mars_route/$DATASET"
OUT="$STAGE/$DATASET"

SPLIT="${REMP_SPLIT:-$R2/protocol/router_dev_smoke100.jsonl}"
if [[ ! -f "$SPLIT" ]]; then
  mkdir -p "$OUT/protocol"
  "$PY" "$STAGE/prepare_remp_splits.py" \
    --dataset "$DATASET" \
    --data-root "$V17/data" \
    --output-dir "$OUT/protocol"
  SPLIT="$OUT/protocol/router_dev_smoke100.jsonl"
fi
INDEX="${REMP_LOCAL_INDEX_ROOT:-$V17/retrieval/local_indexes/$DATASET/topic_silo}"
ASSIGN="${REMP_ASSIGNMENT:-$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl}"
P0="${REMP_P0_CENTROIDS:-$V17/partitions/centroids/$DATASET/topic_silo_m20.npy}"

mkdir -p "$OUT/resource_profiles" "$OUT/run1" "$OUT/run2" "$OUT/reports"

if [[ ! -f "$OUT/resource_profiles/remp_profile_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/build_remp_profiles.py" \
    --dataset "$DATASET" \
    --local-index-root "$INDEX" \
    --p0-centroids "$P0" \
    --output-dir "$OUT/resource_profiles" \
    --device cuda
fi

for RUN in run1 run2; do
  if [[ ! -f "$OUT/$RUN/remp_candidate_gate_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/run_remp_candidate_gate.py" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --profiles "$OUT/resource_profiles/remp_client_profiles.json" \
      --assignment "$ASSIGN" \
      --output-root "$OUT/$RUN" \
      --device cuda
  fi
done

cmp -s "$OUT/run1/candidate_recall.csv" "$OUT/run2/candidate_recall.csv"
cmp -s "$OUT/run1/candidate_recall_per_query.csv" "$OUT/run2/candidate_recall_per_query.csv"

printf '{"stage":"REM-P","dataset":"%s","byte_identical_summary":true,"byte_identical_per_query":true,"reader_started":false,"final_test_accessed":false}\n' "$DATASET" \
  > "$OUT/reports/reproducibility.json"
