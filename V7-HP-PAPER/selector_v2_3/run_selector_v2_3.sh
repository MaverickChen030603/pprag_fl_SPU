#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs/labels outputs/model_cv outputs/calibration outputs/final_1000 outputs/ablation outputs/diagnostics reports
python3 build_positive_action_labels.py
python3 train_answer_neutral_positive_selector.py
python3 calibrate_positive_selector_cv.py
python3 run_answer_neutral_selector_1000.py
python3 run_selector_v2_3_ablation.py
python3 diagnose_selector_v2_3_failures.py
python3 oracle_gap_recall_analysis.py
python3 make_paper_ready_report.py
python3 - <<'PY'
import json
from pathlib import Path
p=Path('outputs/final_1000/final_1000_crossfit_summary.json')
print(json.dumps(json.loads(p.read_text()), ensure_ascii=False, indent=2))
PY
