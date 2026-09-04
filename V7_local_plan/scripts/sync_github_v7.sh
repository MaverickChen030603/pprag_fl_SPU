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

log "Adding V7 automation and reports"
git add V7 run_v7_all.sh check_v7_status.sh v7_all.log 2>/dev/null || git add V7 run_v7_all.sh check_v7_status.sh

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
