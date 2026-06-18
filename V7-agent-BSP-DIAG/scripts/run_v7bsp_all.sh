#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$BASE/scripts/check_fid_reader.sh"
for suite in v7bsp_main v7bsp_bsp_methods v7bsp_memory_ablation; do
  bash "$BASE/scripts/run_v7bsp_suite.sh" "$suite"
done
bash "$BASE/scripts/run_v7bsp_official_fid.sh"
/home/iiserver31/anaconda3/envs/supv2/bin/python "$BASE/scripts/analyze_v7bsp.py"
/home/iiserver31/anaconda3/envs/supv2/bin/python "$BASE/scripts/generate_v7bsp_report.py"
