#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MAX_QUERIES="${MAX_QUERIES:-100}"
ORACLE_QUERIES="${ORACLE_QUERIES:-20}"
export PYTHONPATH="$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/retrieval/phase_a_smoke" "$ROOT/oracle_search/phase_a_smoke"

run_dataset() {
  local dataset="$1" gpu="$2"
  local split="$ROOT/data/$dataset/development.jsonl"
  local index="$ROOT/retrieval/indexes/$dataset.sqlite"
  local output="$ROOT/retrieval/phase_a_smoke/$dataset"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/retrieval/02_generate_frozen_pools.py" \
    --split "$split" --dataset "$dataset" --index "$index" --output-dir "$output" \
    --max-queries "$MAX_QUERIES" --device cuda
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/retrieval/03_score_crossencoder.py" \
    --split "$split" --pool "$output/top10.jsonl" --output "$output/top10_ce.jsonl" --device cuda
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/retrieval/03_score_crossencoder.py" \
    --split "$split" --pool "$output/top20.jsonl" --output "$output/top20_ce.jsonl" --device cuda
  "$PYTHON" "$ROOT/evaluation/01_audit_candidate_pools.py" \
    --split "$split" --pool "$output/top10_ce.jsonl" --dataset "$dataset" --output-dir "$output"
  "$PYTHON" "$ROOT/evaluation/01_audit_candidate_pools.py" \
    --split "$split" --pool "$output/top20_ce.jsonl" --dataset "$dataset" --output-dir "$output"
  "$PYTHON" "$ROOT/oracle_search/01_generate_oracle_trajectories.py" \
    --input "$output/top10_ce.jsonl" --output "$ROOT/oracle_search/phase_a_smoke/${dataset}_top10_contexts.jsonl" \
    --pool-size 10 --beam-width 16 --depth 3 --max-queries "$ORACLE_QUERIES"
}

run_dataset hotpotqa 0 >"$ROOT/logs/phase_a_hotpotqa.log" 2>&1 & p0=$!
run_dataset 2wikimultihopqa 1 >"$ROOT/logs/phase_a_2wiki.log" 2>&1 & p1=$!
run_dataset musique 2 >"$ROOT/logs/phase_a_musique.log" 2>&1 & p2=$!

status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one Phase-A smoke worker failed. Inspect logs/phase_a_*.log." >&2
  exit "$status"
fi

"$PYTHON" "$ROOT/protocol/03_no_leak_audit.py"
"$PYTHON" "$ROOT/protocol/04_validate_frozen_splits.py"
echo "V16 Phase-A retrieval smoke complete. Reader labeling is intentionally a separate frozen step."
