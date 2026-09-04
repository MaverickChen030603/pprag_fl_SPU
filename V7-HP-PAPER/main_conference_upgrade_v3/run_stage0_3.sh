#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDE4RAG_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
V3="$ROOT/V7-HP-PAPER/main_conference_upgrade_v3"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$V3"
"$PYTHON_BIN" 00_main_conference_gap_audit.py
"$PYTHON_BIN" 01_candidate_opportunity_audit.py
"$PYTHON_BIN" 02_generate_reader_compatible_actions.py
"$PYTHON_BIN" 03_run_new_action_reader_outcomes.py --resume "$@"
