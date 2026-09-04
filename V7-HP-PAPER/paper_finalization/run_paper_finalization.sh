#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python3 paper_finalization_common.py all
python3 - <<'PY'
from pathlib import Path
required = [
"outputs/tables/main_result_table.md",
"outputs/tables/selector_evolution_table.md",
"outputs/tables/ablation_table.md",
"outputs/tables/candidate_pool_quality_table.md",
"outputs/tables/positive_feature_importance_table.md",
"outputs/latex/main_result_table.tex",
"outputs/latex/selector_evolution_table.tex",
"outputs/latex/ablation_table.tex",
"outputs/figures/selector_evolution.png",
"outputs/figures/v2_3_metric_delta_bar.png",
"outputs/figures/positive_recall_comparison.png",
"outputs/figures/candidate_pool_breakdown.png",
"outputs/figures/failure_distribution.png",
"outputs/figures/ablation_comparison.png",
"outputs/audit/no_leak_crossfit_audit.md",
"outputs/diagnostics/candidate_pool_quality_summary.json",
"outputs/diagnostics/positive_feature_importance.json",
"outputs/case_studies/success_cases.md",
"outputs/case_studies/answer_neutral_cases.md",
"outputs/case_studies/failure_cases.md",
"reports/experiment_section_draft.md",
"reports/paper_claim_boundary_memo.md",
"reports/paper_finalization_report.md",
]
missing = [p for p in required if not Path(p).exists()]
print("missing:", missing)
if missing:
    raise SystemExit(1)
PY
