#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="${EXPERIMENT_NAME:-pprag_fl_v7_hp1}"

echo "== V7-HP1 process =="
pgrep -af 'V7-HP1|run_v7_hp1|run_hp1_strict_eval|write_hp1_analysis' || true

echo
echo "== Upstream run_metadata count =="
if [[ -d "$ROOT/V7-HP1/outputs/$EXP" ]]; then
  find "$ROOT/V7-HP1/outputs/$EXP" -name run_metadata.json | awk -F/ '{print $(NF-3)}' | sort | uniq -c
  echo "total $(find "$ROOT/V7-HP1/outputs/$EXP" -name run_metadata.json | wc -l)"
else
  echo "none"
fi

echo
echo "== Strict eval metric count =="
if [[ -d "$ROOT/V7-HP1/outputs/hp1_strict_eval" ]]; then
  find "$ROOT/V7-HP1/outputs/hp1_strict_eval" -name hp1_strict_metrics.json | awk -F/ '{print $(NF-3)}' | sort | uniq -c
  echo "summary files $(find "$ROOT/V7-HP1/outputs/hp1_strict_eval" -name hp1_strict_summary.csv | wc -l)"
else
  echo "none"
fi

echo
echo "== Latest reports =="
find "$ROOT/实验分析报告/V7-HP1" -maxdepth 1 -type f -name 'v7_hp1_auto_analysis*.md' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort || true

echo
echo "== Log tail =="
tail -n 40 "$ROOT/v7_hp1_all.log" 2>/dev/null || true
