#!/usr/bin/env bash
# Prepare fresh splits, then run the preregistered two-dataset Recovery-Dev gate.
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
STAGE="$V20/stage_r2a6_resource_memory"
GPU_2WIKI="${REMP_GPU_2WIKI:-0}"
GPU_MUSIQUE="${REMP_GPU_MUSIQUE:-1}"
mkdir -p "$STAGE/protocol" "$STAGE/reports" "$STAGE/efficiency" "$STAGE/error_analysis" "$STAGE/holdout"

for DATASET in 2wikimultihopqa musique; do
  R2="$V20/stage_r2_mars_route/$DATASET"
  mkdir -p "$STAGE/protocol/$DATASET"
  "$PY" "$STAGE/protocol/prepare_recovery_splits.py" --dataset "$DATASET" --r2-root "$R2" --output-dir "$STAGE/protocol/$DATASET"
  "$PY" "$STAGE/protocol/audit_no_leak.py" --dataset "$DATASET" --r2-root "$R2" --stage-root "$STAGE" --recovery-protocol-dir "$STAGE/protocol/$DATASET"
done
"$PY" "$STAGE/protocol/combine_recovery_split_manifests.py" --stage-root "$STAGE"

bash "$STAGE/run_recovery_dev_dataset.sh" 2wikimultihopqa "$GPU_2WIKI" &
PID_2WIKI=$!
bash "$STAGE/run_recovery_dev_dataset.sh" musique "$GPU_MUSIQUE" &
PID_MUSIQUE=$!
wait "$PID_2WIKI"
wait "$PID_MUSIQUE"

"$PY" "$STAGE/protocol/combine_no_leak_audits.py" --stage-root "$STAGE"

"$PY" "$STAGE/reports/aggregate_recovery.py" \
  --phase dev \
  --stage-root "$STAGE" \
  --run-root "$STAGE/candidate_generation/recovery_dev/run1" \
  --profile-root "$STAGE/memory_profiles"
