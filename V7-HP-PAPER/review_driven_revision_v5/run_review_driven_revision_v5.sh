#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
V5="${V5:-$PROJECT_ROOT/V7-HP-PAPER/review_driven_revision_v5}"
PY="${PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
LOG="$V5/logs"
mkdir -p "$LOG"
cd "$PROJECT_ROOT"

run_four_gpu() {
  local label="$1"
  shift
  local pids=()
  for shard in 0 1 2 3; do
    "$PY" -u "$@" --device "cuda:$shard" --shard-id "$shard" --num-shards 4 --resume \
      >"$LOG/${label}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}

echo "[v5] review audit"
"$PY" "$V5/00_review_weakness_audit.py"

if [[ "${SKIP_RECOMP_DEV:-0}" != "1" ]]; then
  echo "[v5] RECOMP development score/build/reader/evaluate"
  run_four_gpu recomp_dev_score "$V5/01_recomp_budget_matched.py" score --split development --batch-size 64
  "$PY" "$V5/01_recomp_budget_matched.py" build --split development
  run_four_gpu recomp_dev_reader "$V5/01_recomp_budget_matched.py" reader --split development --batch-size 24
  "$PY" "$V5/01_recomp_budget_matched.py" evaluate --split development
fi

echo "[v5] RECOMP frozen 3,000 holdout"
run_four_gpu recomp_holdout_score "$V5/01_recomp_budget_matched.py" score --split holdout --batch-size 64
"$PY" "$V5/01_recomp_budget_matched.py" build --split holdout
run_four_gpu recomp_holdout_reader "$V5/01_recomp_budget_matched.py" reader --split holdout --batch-size 24
"$PY" "$V5/01_recomp_budget_matched.py" evaluate --split holdout

if [[ "${SKIP_LITE_DEV:-0}" != "1" ]]; then
  echo "[v5] Lite development and fully nested evaluation"
  "$PY" "$V5/02_build_lite_generator.py" --stage development
  run_four_gpu lite_dev_reader "$V5/03_run_lite_nested_evaluation.py" reader --batch-size 24
  "$PY" "$V5/03_run_lite_nested_evaluation.py" nested
  "$PY" "$V5/03_run_lite_nested_evaluation.py" official
  "$PY" "$V5/03_run_lite_nested_evaluation.py" freeze
fi

echo "[v5] untouched 3,405-query revision holdout"
"$PY" "$V5/02_build_lite_generator.py" --stage revision-holdout
"$PY" "$V5/02_build_lite_generator.py" --stage revision-full-v4 --device cuda:0 --batch-size 64 --reuse-cache
run_four_gpu lite_revision_reader "$V5/03_run_lite_nested_evaluation.py" revision-reader --batch-size 24
"$PY" "$V5/03_run_lite_nested_evaluation.py" revision-official

echo "[v5] 2Wiki frozen-generator safety calibration"
"$PY" "$V5/05_run_2wiki_safety_calibration.py" prepare --device cuda:0 --batch-size 64 --reuse-cache
run_four_gpu wiki2_calibration_reader "$V5/05_run_2wiki_safety_calibration.py" reader --batch-size 24
"$PY" "$V5/05_run_2wiki_safety_calibration.py" calibrate

echo "[v5] cost and bounded-pool scope"
"$PY" "$V5/04_measure_system_cost.py" analyze
"$PY" "$V5/06_run_pool_size_sensitivity.py"
for system in frozen_top5_baseline full_v4 lite_method recomp_top1; do
  "$PY" "$V5/04_measure_system_cost.py" benchmark-reader --system "$system" --device cuda:0 \
    --sample-size 30 --warmup 3 >"$LOG/cost_${system}.log" 2>&1
done
"$PY" "$V5/04_measure_system_cost.py" benchmark-reader --system recomp_budgetmatched --device cuda:0 \
  --sample-size 30 --warmup 3 >"$LOG/cost_recomp_budgetmatched.log" 2>&1
"$PY" "$V5/04_measure_system_cost.py" report

echo "[v5] tables, revised paper, response, and claim audit"
"$PY" "$V5/07_build_review_tables.py"
"$PY" "$V5/08_write_revised_paper.py"

for required in \
  "$V5/outputs/recomp/recomp_holdout_metrics.json" \
  "$V5/outputs/lite_model/lite_holdout_metrics.json" \
  "$V5/outputs/2wiki_calibration/calibration_results.json" \
  "$V5/outputs/cost/runtime_benchmark.json" \
  "$V5/paper/paper_main_conference_v5.md" \
  "$V5/review_response_major_revision.md" \
  "$V5/submission_readiness_report.md"; do
  test -s "$required" || { echo "Missing required artifact: $required" >&2; exit 1; }
done
echo "[v5] complete: $V5"
