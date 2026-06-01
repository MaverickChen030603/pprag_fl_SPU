#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$ROOT/V7-H1/outputs/pprag_fl_v7_h1"
STRICT_ROOT="$ROOT/V7-H1/outputs/h1_strict_eval"
LOG="${LOG:-$ROOT/v7_h1_all.log}"

echo "== V7-H1 process =="
pgrep -af 'run_v7_h1_all|V7-H1/run_experiment_suite|run_h1_strict_eval' || true

echo
echo "== Upstream run_metadata count =="
if [[ -d "$EXP_ROOT" ]]; then
  find "$EXP_ROOT" -name run_metadata.json | sed "s#^$EXP_ROOT/##" | awk -F/ '{print $1}' | sort | uniq -c | sort
  echo "total $(find "$EXP_ROOT" -name run_metadata.json | wc -l | tr -d ' ')"
else
  echo "no upstream root yet: $EXP_ROOT"
fi

echo
echo "== Strict eval metric count =="
if [[ -d "$STRICT_ROOT" ]]; then
  find "$STRICT_ROOT" -name h1_strict_metrics.json | sed "s#^$STRICT_ROOT/##" | awk -F/ '{print $1}' | sort | uniq -c | sort
  echo "summary files $(find "$STRICT_ROOT" -name h1_strict_summary.csv | wc -l | tr -d ' ')"
else
  echo "no strict root yet: $STRICT_ROOT"
fi

echo
echo "== Latest reports =="
find "$ROOT/实验分析报告/V7-H1" -maxdepth 1 -type f -name '*.md' -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -5 || true

echo
echo "== Log tail =="
if [[ -f "$LOG" ]]; then
  tail -40 "$LOG"
else
  echo "no log yet: $LOG"
fi
