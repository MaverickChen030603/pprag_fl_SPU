#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
BRANCH="${BRANCH:-main}"
MESSAGE="${MESSAGE:-Add V7 agentic federated RAG experiment automation}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

cd "$ROOT"

log "Git status before sync"
git status --short

log "Adding V7 code, configs, automation, and design documents"
find V7 \
  -path 'V7/outputs' -prune -o \
  -path 'V7/outputs/*' -prune -o \
  -path 'V7/outputs_bootstrap_from_v6_*' -prune -o \
  -path 'V7/outputs_bootstrap_from_v6_*/*' -prune -o \
  -path '*/__pycache__' -prune -o \
  -path '*/__pycache__/*' -prune -o \
  -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) \
  -print0 | xargs -0 -r git add
git add run_v7_all.sh check_v7_status.sh sync_github_v7.sh

if git diff --cached --quiet; then
  log "No staged changes to commit"
else
  git commit -m "$MESSAGE"
fi

log "Pulling remote branch before push"
git pull --rebase origin "$BRANCH"

log "Pushing to origin/$BRANCH"
git push origin "$BRANCH"

log "GitHub sync complete"
