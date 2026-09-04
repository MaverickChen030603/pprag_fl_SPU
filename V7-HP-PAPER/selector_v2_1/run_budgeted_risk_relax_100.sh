#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" V7-HP-PAPER/selector_v2_1/run_budgeted_risk_relax_100.py
