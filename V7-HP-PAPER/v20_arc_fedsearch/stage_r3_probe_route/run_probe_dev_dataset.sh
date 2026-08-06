#!/usr/bin/env bash
# Run one deterministic, reader-free R3 Probe-Dev audit.
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

mkdir -p "$STAGE/protocol/$DATASET" "$STAGE/runs"
"$PY" "$STAGE/protocol/prepare_probe_splits.py" \
  --dataset "$DATASET" --r2-root "$R2" --output-dir "$STAGE/protocol/$DATASET"

for RUN in run1 run2; do
  OUT="$STAGE/runs/$RUN/$DATASET"
  if [[ ! -f "$OUT/reports/probe_route_go_no_go.json" ]]; then
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/run_probe_audit.py" \
      --dataset "$DATASET" \
      --split "$STAGE/protocol/$DATASET/probe_dev.jsonl" \
      --profiles "$R2/resource_profiles/client_profiles.json" \
      --assignment "$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl" \
      --local-index-root "$V17/retrieval/local_indexes/$DATASET/topic_silo" \
      --output-root "$OUT" --device cuda
  fi
done

"$PY" "$STAGE/verify_reproducibility.py" \
  --dataset "$DATASET" \
  --run1 "$STAGE/runs/run1/$DATASET" \
  --run2 "$STAGE/runs/run2/$DATASET" \
  --output "$STAGE/protocol/$DATASET/reproducibility.json"

for AREA in probe_features probe_oracle label_free_baselines reports; do
  mkdir -p "$STAGE/$AREA/$DATASET"
  cp -a "$STAGE/runs/run1/$DATASET/$AREA/." "$STAGE/$AREA/$DATASET/"
done
cp -a "$STAGE/runs/run1/$DATASET/protocol_no_leak_audit.json" "$STAGE/protocol/$DATASET/no_leak_audit.json"
