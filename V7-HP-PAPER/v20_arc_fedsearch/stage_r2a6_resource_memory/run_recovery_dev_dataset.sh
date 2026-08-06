#!/usr/bin/env bash
# Run one fixed R2-A.6 Recovery-Dev dataset without reader/router training.
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
STAGE="$V20/stage_r2a6_resource_memory"
R2="$V20/stage_r2_mars_route/$DATASET"
PROFILE="$STAGE/memory_profiles/$DATASET"
RUNROOT="$STAGE/candidate_generation/recovery_dev"
BEST_P=8
[[ "$DATASET" == "musique" ]] && BEST_P=16

mkdir -p "$PROFILE" "$RUNROOT/run1/$DATASET" "$RUNROOT/run2/$DATASET"
if [[ ! -f "$PROFILE/profile_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/memory_profiles/build_remp_profiles.py" \
    --dataset "$DATASET" \
    --router-train "$R2/protocol/router_train.jsonl" \
    --local-index-root "$V17/retrieval/local_indexes/$DATASET/topic_silo" \
    --output-dir "$PROFILE" \
    --device cuda \
    --seed 20260806
fi

"$PY" "$STAGE/protocol/audit_no_leak.py" \
  --dataset "$DATASET" \
  --r2-root "$R2" \
  --stage-root "$STAGE" \
  --recovery-protocol-dir "$STAGE/protocol/$DATASET" \
  --units "$PROFILE/client_memory_units.jsonl" \
  --train-corpus-manifest "$PROFILE/train_corpus_manifest.json"

for RUN in run1 run2; do
  OUT="$RUNROOT/$RUN/$DATASET"
  if [[ ! -f "$OUT/candidate_run_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/candidate_generation/evaluate_recovery.py" \
      --dataset "$DATASET" \
      --split "$STAGE/protocol/$DATASET/recovery_dev.jsonl" \
      --legacy-profiles "$R2/resource_profiles/client_profiles.json" \
      --units "$PROFILE/client_memory_units.jsonl" \
      --selected-embeddings "$PROFILE/selected_unit_embeddings.npz" \
      --assignment "$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl" \
      --output-dir "$OUT" \
      --best-p "$BEST_P" \
      --device cuda
  fi
done

cmp -s "$RUNROOT/run1/$DATASET/recovery_results_per_query.csv" "$RUNROOT/run2/$DATASET/recovery_results_per_query.csv"
printf '{"dataset":"%s","two_runs_byte_identical":true,"reader_started":false,"final_test_accessed":false}\n' "$DATASET" > "$PROFILE/reproducibility.json"
