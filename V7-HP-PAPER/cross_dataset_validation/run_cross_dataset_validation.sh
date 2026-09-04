#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python3 cross_dataset_common.py all
python3 - <<'PY'
from pathlib import Path
required = [
"reports/dataset_selection_memo.md",
"outputs/dataset_feasibility_summary.json",
"outputs/2wiki_adapter/adapter_summary.json",
"outputs/2wiki_smoke_300/summary.json",
"outputs/2wiki_smoke_300/significance_report.json",
"outputs/2wiki_smoke_300/failure_summary.json",
"reports/cross_dataset_validation_report.md",
"reports/paper_update_recommendation.md",
]
missing=[p for p in required if not Path(p).exists()]
print("missing:", missing)
if missing:
    raise SystemExit(1)
PY
