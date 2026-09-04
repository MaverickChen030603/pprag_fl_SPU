#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 <2wikimultihopqa|musique> <cuda-device>" >&2; exit 2; fi
DATASET="$1"; DEVICE="$2"
case "$DATASET" in 2wikimultihopqa|musique) ;; *) exit 2 ;; esac
ROOT=/home/iiserver31/projects/FedE4RAG-main
PY=/home/iiserver31/anaconda3/envs/supv2/bin/python
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
STAGE="$V20/stage_lr_memory_local_recovery"
B0="$V20/stage_r2b0_subset_attainability/$DATASET/run1"
OUT="$STAGE/$DATASET"
ATTEMPT="${LR_ATTEMPT:-}"
MAX="$(printenv LR_MAX_QUERIES || true)"
if [[ -z "$MAX" ]]; then MAX=0; fi
if [[ "$MAX" != "0" ]]; then OUT="$OUT/dryrun_$MAX"; fi
if [[ -n "$ATTEMPT" ]]; then
  [[ "$ATTEMPT" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid LR_ATTEMPT: $ATTEMPT" >&2; exit 2; }
  OUT="$OUT/$ATTEMPT"
fi
if [[ -e "$OUT/reports/reproducibility.json" ]]; then echo "refusing to overwrite $OUT" >&2; exit 4; fi
if [[ -d "$OUT/run1" || -d "$OUT/run2" ]]; then
  echo "refusing to overwrite incomplete output $OUT; choose a new LR_ATTEMPT" >&2
  exit 4
fi
SPLIT="$V20/stage_r2_mars_route/$DATASET/protocol/router_dev_smoke100.jsonl"
PROFILES="$V20/stage_remp_evidence_memory/$DATASET/resource_profiles/remp_client_profiles.json"
ASSIGN="$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl"
INDEX="$V17/retrieval/local_indexes/$DATASET/topic_silo"
for INPUT in "$SPLIT" "$PROFILES" "$ASSIGN" "$B0/inference/frozen_candidates.csv" "$B0/evaluation/per_query_selected_metrics.csv" "$B0/frozen_retrieval/frozen_subset_retrieval.csv"; do
  [[ -f "$INPUT" ]] || { echo "missing LR-0 input: $INPUT" >&2; exit 3; }
done
mkdir -p "$OUT/protocol" "$OUT/run1" "$OUT/run2" "$OUT/reports"
printf '{"stage":"LR-0","dataset":"%s","attempt":"%s","max_queries":%s,"split_sha256":"%s","profiles_sha256":"%s","reader_started":false,"final_test_accessed":false}\n' \
  "$DATASET" "$ATTEMPT" "$MAX" "$(sha256sum "$SPLIT" | awk '{print $1}')" "$(sha256sum "$PROFILES" | awk '{print $1}')" > "$OUT/protocol/input_manifest.json"
for RUN in run1 run2; do
  RUN_OUT="$OUT/$RUN"
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/build_lr_inference.py" --dataset "$DATASET" --phase lr0 --split "$SPLIT" \
    --b0-candidates "$B0/inference/frozen_candidates.csv" --b0-oracle "$B0/evaluation/per_query_selected_metrics.csv" \
    --profiles "$PROFILES" --local-index-root "$INDEX" --output-root "$RUN_OUT/inference" --max-queries "$MAX" --device cuda
  "$PY" "$STAGE/freeze_lr_outputs.py" --selections "$RUN_OUT/inference/track_selections.jsonl" \
    --rankings "$RUN_OUT/inference/local_rankings_lr0.jsonl" --output-root "$RUN_OUT/frozen" --methods L0
  "$PY" "$STAGE/evaluate_lr.py" --dataset "$DATASET" --split "$SPLIT" --assignment "$ASSIGN" --local-index-root "$INDEX" \
    --frozen-outputs "$RUN_OUT/frozen/frozen_lr_outputs.csv" --output-root "$RUN_OUT/evaluation" --b0-frozen "$B0/frozen_retrieval/frozen_subset_retrieval.csv"
done
cmp -s "$OUT/run1/inference/track_selections.jsonl" "$OUT/run2/inference/track_selections.jsonl"
cmp -s "$OUT/run1/inference/local_rankings_lr0.jsonl" "$OUT/run2/inference/local_rankings_lr0.jsonl"
cmp -s "$OUT/run1/frozen/frozen_lr_outputs.csv" "$OUT/run2/frozen/frozen_lr_outputs.csv"
cmp -s "$OUT/run1/evaluation/per_query_metrics.csv" "$OUT/run2/evaluation/per_query_metrics.csv"
cmp -s "$OUT/run1/evaluation/summary.csv" "$OUT/run2/evaluation/summary.csv"
cp "$OUT/run1/evaluation/lr0_decision.json" "$OUT/reports/lr0_decision.json"
printf '{"stage":"LR-0","dataset":"%s","byte_identical_selections":true,"byte_identical_rankings":true,"byte_identical_frozen_outputs":true,"byte_identical_per_query":true,"byte_identical_summary":true,"reader_started":false,"final_test_accessed":false}\n' "$DATASET" > "$OUT/reports/reproducibility.json"
