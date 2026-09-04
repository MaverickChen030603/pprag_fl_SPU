#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$ROOT/logs/frozen_cost_benchmark.done" ]]; then
  "$ROOT/run_frozen_cost_benchmark.sh"
fi

"$PYTHON_BIN" "$ROOT/build_final_evidence.py"
"$PYTHON_BIN" "$ROOT/write_final_submission.py"
"$PYTHON_BIN" "$ROOT/run_final_audits.py"

printf 'complete\n' > "$ROOT/logs/final_submission.done"
