#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V15="$ROOT/../v15_robust_context_repair"
V16="$ROOT/../v16_action_composition"
INDEX_ROOT="$ROOT/retrieval/local_indexes"
mkdir -p "$ROOT/logs" "$INDEX_ROOT"

source_index() {
  case "$1" in
    hotpotqa) echo "$V15/retrieval/indexes/hotpotqa.sqlite" ;;
    2wikimultihopqa) echo "$V15/retrieval/indexes/2wikimultihopqa.sqlite" ;;
    musique) echo "$V16/retrieval/indexes/musique.sqlite" ;;
  esac
}

build_dataset() {
  local dataset="$1" partition
  for partition in topic_silo entity_community random_control; do
    "$PYTHON" "$ROOT/retrieval/00_build_client_local_indexes.py" \
      --dataset "$dataset" --partition "$partition" \
      --source-index "$(source_index "$dataset")" \
      --assignment "$ROOT/partitions/assignments/$dataset/${partition}_m20.jsonl" \
      --output-root "$INDEX_ROOT" --m 20
  done
}

build_dataset hotpotqa >"$ROOT/logs/local_index_hotpotqa.log" 2>&1 & p0=$!
build_dataset 2wikimultihopqa >"$ROOT/logs/local_index_2wikimultihopqa.log" 2>&1 & p1=$!
build_dataset musique >"$ROOT/logs/local_index_musique.log" 2>&1 & p2=$!

status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one local-index builder failed; inspect logs/local_index_*.log" >&2
  exit "$status"
fi
echo "V17 physical client-local indexes complete."
