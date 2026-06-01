#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESSAGE="${MESSAGE:-Add V7-H1 hard tail agent gap experiment}"
cd "$ROOT"

git pull --rebase --autostash origin main

git add \
  run_v7_h1_all.sh \
  check_v7_h1_status.sh \
  sync_github_v7_h1.sh \
  V7-H1/experiment_config.py \
  V7-H1/upload_selectors.py \
  V7-H1/fedrag_selective_upload.py \
  V7-H1/run_upstream.py \
  V7-H1/run_experiment_suite.py \
  V7-H1/run_h1_strict_eval.py \
  V7-H1/write_h1_analysis.py \
  V7-H1/V7_H1_experiment_plan_cn.md

git reset -q -- V7-H1/outputs V7-H1/__pycache__ 2>/dev/null || true

if git diff --cached --quiet; then
  echo "No staged V7-H1 changes to commit."
else
  git commit -m "$MESSAGE"
fi

git push origin main
