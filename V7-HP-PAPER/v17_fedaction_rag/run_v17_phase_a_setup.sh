#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V15="$ROOT/../v15_robust_context_repair"
V16="$ROOT/../v16_action_composition"
mkdir -p "$ROOT/logs" "$ROOT/partitions/assignments" "$ROOT/partitions/centroids"

partition_dataset() {
  local dataset="$1" index="$2" gpu="$3"
  local marker="$ROOT/logs/partition_${dataset}.done"
  if [[ -f "$marker" ]]; then
    echo "$dataset partition already complete"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/partitions/01_build_client_partitions.py" \
    --dataset "$dataset" --index "$index" --output-root "$ROOT/partitions" \
    --m 20 --seed 20260723 --device cuda --batch-size 256
  touch "$marker"
}

partition_dataset hotpotqa "$V15/retrieval/indexes/hotpotqa.sqlite" 0 \
  >"$ROOT/logs/partition_hotpotqa.log" 2>&1 & p0=$!
partition_dataset 2wikimultihopqa "$V15/retrieval/indexes/2wikimultihopqa.sqlite" 1 \
  >"$ROOT/logs/partition_2wikimultihopqa.log" 2>&1 & p1=$!
partition_dataset musique "$V16/retrieval/indexes/musique.sqlite" 2 \
  >"$ROOT/logs/partition_musique.log" 2>&1 & p2=$!

status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one partition worker failed; inspect logs/partition_*.log" >&2
  exit "$status"
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/partitions/02_assign_query_origins.py" \
  --partition-root "$ROOT/partitions" --data-root "$ROOT/data" --device cuda \
  >"$ROOT/logs/query_origins.log" 2>&1

"$PYTHON" "$ROOT/partitions/03_summarize_non_iid.py" \
  --partition-root "$ROOT/partitions" \
  --query-origins "$ROOT/partitions/client_query_distribution.csv" \
  >"$ROOT/logs/non_iid_statistics.log" 2>&1

"$PYTHON" "$ROOT/oracle/01_evidence_dispersion_audit.py" \
  --partition-root "$ROOT/partitions" --data-root "$ROOT/data" --split development \
  >"$ROOT/logs/evidence_dispersion.log" 2>&1

"$PYTHON" "$ROOT/protocol/03_no_leak_audit.py"
"$PYTHON" "$ROOT/protocol/04_validate_frozen_splits.py"
echo "V17 Phase-A partition, origin, and evidence-dispersion setup complete."
