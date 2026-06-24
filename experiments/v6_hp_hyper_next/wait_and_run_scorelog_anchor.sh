#!/usr/bin/env bash
set -euo pipefail
cd /home/iiserver31/projects/FedE4RAG-main
while true; do
  date '+[queue-check] %F %T %Z checking GPU free memory'
  GPU_ID_CANDIDATE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 >= 38000) {print $1; exit}}')
  if [[ -n "${GPU_ID_CANDIDATE:-}" ]]; then
    date '+[queue-start] %F %T %Z'
    echo "GPU_ID=$GPU_ID_CANDIDATE"
    GPU_ID="$GPU_ID_CANDIDATE" bash experiments/v6_hp_hyper_next/run_scorelog_anchor_hard1000.sh
    date '+[queue-done] %F %T %Z'
    exit 0
  fi
  sleep 300
done
