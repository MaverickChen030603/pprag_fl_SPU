#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
cd "$ROOT"

echo "== processes =="
pgrep -af 'run_v7_all.sh|python V7/run_experiment_suite.py|python V7/finalize_pipeline.py|python V7/run_all_rag_eval.py' || true

echo
echo "== run counts =="
printf 'run_metadata.json: '
find V7/outputs -name run_metadata.json 2>/dev/null | wc -l
printf 'rag_eval_stdout.log: '
find V7/outputs -name rag_eval_stdout.log 2>/dev/null | wc -l

echo
echo "== V7 reports =="
find 实验分析报告/V7 -maxdepth 2 -type d 2>/dev/null | sort | tail -30 || true

echo
echo "== latest log =="
tail -n 80 v7_all.log 2>/dev/null || true
