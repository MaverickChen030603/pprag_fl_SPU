#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$BASE/scripts/check_fid_reader.sh"
for suite in v7pm_main v7pm_memory_ablation v7pm_dynamic_ablation v7pm_bandit_slot; do
  bash "$BASE/scripts/run_v7pm_suite.sh" "$suite"
done
bash "$BASE/scripts/run_v7pm_official_fid.sh"
/home/iiserver31/anaconda3/envs/supv2/bin/python "$BASE/scripts/analyze_v7pm.py"
/home/iiserver31/anaconda3/envs/supv2/bin/python "$BASE/scripts/generate_v7pm_report.py"
