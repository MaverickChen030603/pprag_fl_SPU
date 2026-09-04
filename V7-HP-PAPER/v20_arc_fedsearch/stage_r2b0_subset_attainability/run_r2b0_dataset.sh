#!/usr/bin/env bash
# R2-B0: freeze candidates and all Bc=3 retrieval subsets before gold evaluation.
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
STAGE="$V20/stage_r2b0_subset_attainability"
REMP="$V20/stage_remp_evidence_memory/$DATASET"
R2="$V20/stage_r2_mars_route/$DATASET"
OUT="$STAGE/$DATASET"
MAX_QUERIES="${R2B0_MAX_QUERIES:-0}"
SPLIT="$R2/protocol/router_dev_smoke100.jsonl"
PROFILES="$REMP/resource_profiles/remp_client_profiles.json"
ASSIGNMENT="$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl"
POOL="$V20/stage_r2a5_candidate_compression/$DATASET/local_pool/all_client_rankers_depth10.jsonl"
POOL_MANIFEST="${POOL%.jsonl}.manifest.json"

if [[ ! -f "$SPLIT" || ! -f "$PROFILES" || ! -f "$ASSIGNMENT" || ! -f "$POOL_MANIFEST" ]]; then
  echo "missing frozen R2-B0 input for $DATASET" >&2
  exit 3
fi
if [[ "$MAX_QUERIES" != "0" ]]; then
  OUT="$OUT/dryrun_${MAX_QUERIES}"
fi
if [[ -e "$OUT/reports/reproducibility.json" ]]; then
  echo "refusing to overwrite completed R2-B0 output: $OUT" >&2
  exit 4
fi

mkdir -p "$OUT/protocol" "$OUT/run1" "$OUT/run2" "$OUT/reports"
SPLIT_SHA256="$(sha256sum "$SPLIT" | awk '{print $1}')"
PROFILE_SHA256="$(sha256sum "$PROFILES" | awk '{print $1}')"
POOL_SHA256="$(sha256sum "$POOL" | awk '{print $1}')"
QUERY_COUNT="$(wc -l < "$SPLIT" | tr -d ' ')"
if [[ "$MAX_QUERIES" != "0" ]]; then QUERY_COUNT="$MAX_QUERIES"; fi
GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf 'unavailable')"
printf '{"stage":"R2-B0","dataset":"%s","split":"%s","split_sha256":"%s","profiles_sha256":"%s","local_pool_sha256":"%s","queries":%s,"max_queries":%s,"git_commit":"%s","reader_started":false,"final_test_accessed":false}\n' \
  "$DATASET" "$SPLIT" "$SPLIT_SHA256" "$PROFILE_SHA256" "$POOL_SHA256" "$QUERY_COUNT" "$MAX_QUERIES" "$GIT_COMMIT" \
  > "$OUT/protocol/input_manifest.json"

for RUN in run1 run2; do
  RUN_OUT="$OUT/$RUN"
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/build_frozen_candidates.py" \
    --dataset "$DATASET" --split "$SPLIT" --profiles "$PROFILES" --output-root "$RUN_OUT/inference" \
    --max-queries "$MAX_QUERIES" --device cuda
  "$PY" "$STAGE/freeze_subset_retrieval.py" \
    --candidates "$RUN_OUT/inference/frozen_candidates.csv" --local-pool "$POOL" --output-root "$RUN_OUT/frozen_retrieval"
  "$PY" "$STAGE/evaluate_subset_attainability.py" \
    --dataset "$DATASET" --split "$SPLIT" --assignment "$ASSIGNMENT" \
    --frozen-retrieval "$RUN_OUT/frozen_retrieval/frozen_subset_retrieval.csv" --output-root "$RUN_OUT/evaluation"
done

cmp -s "$OUT/run1/inference/frozen_candidates.csv" "$OUT/run2/inference/frozen_candidates.csv"
cmp -s "$OUT/run1/frozen_retrieval/frozen_subset_retrieval.csv" "$OUT/run2/frozen_retrieval/frozen_subset_retrieval.csv"
cmp -s "$OUT/run1/evaluation/per_query_selected_metrics.csv" "$OUT/run2/evaluation/per_query_selected_metrics.csv"
cmp -s "$OUT/run1/evaluation/summary.csv" "$OUT/run2/evaluation/summary.csv"
cp "$OUT/run1/evaluation/gate_inputs.json" "$OUT/reports/gate_inputs.json"
printf '{"stage":"R2-B0","dataset":"%s","byte_identical_candidates":true,"byte_identical_frozen_retrieval":true,"byte_identical_per_query":true,"byte_identical_summary":true,"reader_started":false,"final_test_accessed":false}\n' "$DATASET" > "$OUT/reports/reproducibility.json"
