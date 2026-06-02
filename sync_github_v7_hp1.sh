#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Stage V7-HP1 code/docs only; never stage experiment outputs or logs.
find V7-HP1 -type f \
  ! -path 'V7-HP1/outputs/*' \
  ! -path 'V7-HP1/outputs_*/*' \
  ! -path 'V7-HP1/__pycache__/*' \
  ! -name '*.pyc' \
  -print0 | xargs -0 git add

git add run_v7_hp1_all.sh check_v7_hp1_status.sh sync_github_v7_hp1.sh

git status --short
git commit -m "Add V7-HP1 HotpotQA agent experiment" || true
git push
