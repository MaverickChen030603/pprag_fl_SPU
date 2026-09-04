#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDE4RAG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
V3="$ROOT/V7-HP-PAPER/main_conference_upgrade_v3"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$V3"
"$PYTHON_BIN" 00_main_conference_gap_audit.py
"$PYTHON_BIN" 01_candidate_opportunity_audit.py
"$PYTHON_BIN" 02_generate_reader_compatible_actions.py
"$PYTHON_BIN" 03_run_new_action_reader_outcomes.py --resume --batch-size "${V3_READER_BATCH_SIZE:-16}"
"$PYTHON_BIN" 04_train_nested_selector_v3.py
"$PYTHON_BIN" 05_run_official_support_evaluation.py
"$PYTHON_BIN" 06_run_multi_reader_evaluation.py --resume --batch-size "${V3_READER_BATCH_SIZE:-16}"
"$PYTHON_BIN" 07_run_scaleup_evaluation.py
"$PYTHON_BIN" 08_run_external_dataset_gate.py
"$PYTHON_BIN" 09_build_main_conference_tables.py
"$PYTHON_BIN" 10_write_main_conference_paper_v3.py
