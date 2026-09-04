#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDE4RAG_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
V4="$ROOT/V7-HP-PAPER/opportunity_aware_semantic_generation_v4"
PYTHON="${V4_PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
GENERATOR_DEVICE="${V4_GENERATOR_DEVICE:-cuda:0}"
FLAN_DEVICE="${V4_FLAN_DEVICE:-cuda:1}"
UNIFIED_DEVICE="${V4_UNIFIED_DEVICE:-cuda:2}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$V4:${PYTHONPATH:-}"

"$PYTHON" "$V4/13_build_same_source_scaleup_contexts.py"
"$PYTHON" "$V4/14_generate_frozen_scaleup_actions.py" --device "$GENERATOR_DEVICE" --batch-size 64 --reuse-cache
"$PYTHON" "$V4/15_apply_frozen_selector_scaleup.py"

"$PYTHON" "$V4/16_run_frozen_scaleup_reader.py" --reader flan --device "$FLAN_DEVICE" --batch-size 16 --resume &
FLAN_PID=$!
"$PYTHON" "$V4/16_run_frozen_scaleup_reader.py" --reader unifiedqa --device "$UNIFIED_DEVICE" --batch-size 16 --resume &
UNIFIED_PID=$!
wait "$FLAN_PID"
wait "$UNIFIED_PID"

"$PYTHON" "$V4/17_run_frozen_scaleup_official_metrics.py"
"$PYTHON" "$V4/18_write_frozen_scaleup_report.py"

echo "Scale-up complete: $V4/reports/same_source_3000_frozen_scaleup_report_cn.md"
