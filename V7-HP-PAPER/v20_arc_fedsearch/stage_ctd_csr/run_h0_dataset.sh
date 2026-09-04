#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then echo "usage: $0 <2wikimultihopqa|musique> <cuda-device>" >&2; exit 2; fi
DATASET="$1"; DEVICE="$2"
case "$DATASET" in 2wikimultihopqa|musique) ;; *) exit 2 ;; esac
ROOT=/home/iiserver31/projects/FedE4RAG-main
PY=/home/iiserver31/anaconda3/envs/supv2/bin/python
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
STAGE="$V20/stage_ctd_csr"
OUT="$STAGE/h0/$DATASET"
MAX="$(printenv H0_MAX_QUERIES || true)"
[[ -n "$MAX" ]] || MAX=0
if [[ "$MAX" != "0" ]]; then OUT="$OUT/dryrun_$MAX"; fi
ATTEMPT="${H0_ATTEMPT:-}"
if [[ -n "$ATTEMPT" ]]; then
  [[ "$ATTEMPT" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid H0_ATTEMPT: $ATTEMPT" >&2; exit 2; }
  OUT="$OUT/$ATTEMPT"
fi
if [[ -e "$OUT/reports/reproducibility.json" || -d "$OUT/run1" || -d "$OUT/run2" ]]; then
  echo "refusing to overwrite H0 output $OUT" >&2; exit 4
fi
SPLIT="$STAGE/h0/protocol/$DATASET/fresh_router_holdout.jsonl"
PROFILES="$V20/stage_remp_evidence_memory/$DATASET/resource_profiles/remp_client_profiles.json"
ASSIGN="$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl"
for INPUT in "$SPLIT" "$PROFILES" "$ASSIGN"; do [[ -f "$INPUT" ]] || { echo "missing H0 input: $INPUT" >&2; exit 3; }; done
mkdir -p "$OUT/protocol" "$OUT/reports"
printf '{"stage":"CTD-CSR-H0","dataset":"%s","max_queries":%s,"split_sha256":"%s","profile_sha256":"%s","reader_started":false,"final_test_accessed":false}\n' \
  "$DATASET" "$MAX" "$(sha256sum "$SPLIT" | awk '{print $1}')" "$(sha256sum "$PROFILES" | awk '{print $1}')" > "$OUT/protocol/input_manifest.json"
for RUN in run1 run2; do
  RUN_OUT="$OUT/$RUN"
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$STAGE/h0_candidate_inference.py" --dataset "$DATASET" --split "$SPLIT" --profiles "$PROFILES" --output-root "$RUN_OUT/inference" --max-queries "$MAX" --device cuda
  "$PY" "$STAGE/h0_candidate_evaluate.py" --dataset "$DATASET" --split "$SPLIT" --assignment "$ASSIGN" --rankings "$RUN_OUT/inference/candidate_rankings.jsonl" --timing "$RUN_OUT/inference/candidate_timing.jsonl" --profiles "$PROFILES" --output-root "$RUN_OUT/evaluation" --max-queries "$MAX"
done

# Timing telemetry is intentionally measured independently on each run, so it
# cannot participate in an exact candidate-output reproducibility check.
normalize_csv_without_telemetry() {
  "$PY" - "$@" <<'PY'
import csv
import sys

input_path, *telemetry_columns = sys.argv[1:]
telemetry_columns = set(telemetry_columns)
with open(input_path, newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = [name for name in reader.fieldnames if name not in telemetry_columns]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        writer.writerow({name: row[name] for name in fieldnames})
PY
}

cmp -s "$OUT/run1/inference/candidate_rankings.jsonl" "$OUT/run2/inference/candidate_rankings.jsonl"
cmp -s "$OUT/run1/inference/query_views.jsonl" "$OUT/run2/inference/query_views.jsonl"
cmp -s <(normalize_csv_without_telemetry "$OUT/run1/evaluation/candidate_per_query.csv" candidate_inference_elapsed_ms) \
  <(normalize_csv_without_telemetry "$OUT/run2/evaluation/candidate_per_query.csv" candidate_inference_elapsed_ms)
cmp -s <(normalize_csv_without_telemetry "$OUT/run1/evaluation/candidate_summary.csv" candidate_query_latency_mean_ms candidate_query_latency_p95_ms) \
  <(normalize_csv_without_telemetry "$OUT/run2/evaluation/candidate_summary.csv" candidate_query_latency_mean_ms candidate_query_latency_p95_ms)
printf '{"stage":"CTD-CSR-H0","dataset":"%s","byte_identical_rankings":true,"byte_identical_query_views":true,"byte_identical_per_query_excluding_latency":true,"byte_identical_summary_excluding_latency":true,"latency_telemetry_excluded_from_identity_check":true,"reader_started":false,"final_test_accessed":false}\n' "$DATASET" > "$OUT/reports/reproducibility.json"
