#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESSAGE="${MESSAGE:-Add complete V7-H1 experiment package}"
cd "$ROOT"

git pull --rebase --autostash origin main

git add run_v7_h1_all.sh check_v7_h1_status.sh sync_github_v7_h1.sh
find V7-H1 -maxdepth 1 -type f \
  \( -name '*.py' -o -name '*.md' -o -name '*.yaml' -o -name 'requirements*.txt' -o -name '__init__.py' \) \
  -print0 | xargs -0 git add

git reset -q -- V7-H1/outputs V7-H1/outputs_bootstrap_from_v6_20260527_152209 V7-H1/__pycache__ 2>/dev/null || true

if git diff --cached --quiet; then
  echo "No staged V7-H1 changes to commit."
else
  git commit -m "$MESSAGE"
fi

git push origin main
