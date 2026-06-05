#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== V7-agent processes ==="
ps -u "$USER" -o pid,ppid,etime,stat,pcpu,pmem,cmd --sort=-pcpu | grep -E 'V7-agent|v7_agent|pprag_fl_v7_agent' | grep -v grep || true
echo
echo "=== recent log ==="
tail -80 "$ROOT/v7_agent_all.log" 2>/dev/null || true
echo
echo "=== strict summaries ==="
find "$ROOT/V7-agent/outputs/hp1_strict_eval" -name hp1_strict_summary.csv -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort -r | head -20 || true
