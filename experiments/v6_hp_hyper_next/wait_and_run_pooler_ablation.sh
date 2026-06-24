#!/usr/bin/env bash
set -euo pipefail

cd /home/iiserver31/projects/FedE4RAG-main

while true; do
  date '+[pooler-queue-check] %F %T %Z checking GPU free memory'
  GPU_ID_CANDIDATE=$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 >= 38000) {print $1; exit}}'
  )
  if [[ -n "${GPU_ID_CANDIDATE:-}" ]]; then
    date '+[pooler-queue-start] %F %T %Z'
    echo "GPU_ID=$GPU_ID_CANDIDATE"
    GROUP=pooler GPU_ID="$GPU_ID_CANDIDATE" bash experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh
    date '+[pooler-queue-done] %F %T %Z'
    exit 0
  fi
  sleep 300
done
