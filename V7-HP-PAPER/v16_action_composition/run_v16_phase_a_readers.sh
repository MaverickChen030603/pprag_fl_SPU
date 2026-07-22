#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MAX_CONTEXTS="${MAX_CONTEXTS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
export PYTHONPATH="$ROOT"
mkdir -p "$ROOT/multi_reader/phase_a_smoke" "$ROOT/logs" "$ROOT/oracle_search/phase_a_smoke/results"

run_reader() {
  local dataset="$1" reader="$2" model="$3" gpu="$4"
  local output="$ROOT/multi_reader/phase_a_smoke/${dataset}_${reader}.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/multi_reader/01_label_oracle_contexts.py" \
    --reader "$reader" --model "$model" --dataset "$dataset" \
    --split "$ROOT/data/$dataset/development.jsonl" \
    --pool "$ROOT/retrieval/phase_a_smoke/$dataset/top10_ce.jsonl" \
    --contexts "$ROOT/oracle_search/phase_a_smoke/${dataset}_top10_contexts.jsonl" \
    --support-predictor "$ROOT/evaluation/checkpoints/${dataset}_support.joblib" \
    --output "$output" --device cuda --batch-size "$BATCH_SIZE" \
    --max-contexts-per-query "$MAX_CONTEXTS" --resume
}

run_reader hotpotqa flan google/flan-t5-large 0 >"$ROOT/logs/reader_hotpot_flan.log" 2>&1 & p0=$!
run_reader 2wikimultihopqa flan google/flan-t5-large 1 >"$ROOT/logs/reader_2wiki_flan.log" 2>&1 & p1=$!
run_reader musique flan google/flan-t5-large 2 >"$ROOT/logs/reader_musique_flan.log" 2>&1 & p2=$!
run_reader hotpotqa unifiedqa allenai/unifiedqa-v2-t5-large-1363200 3 >"$ROOT/logs/reader_hotpot_unifiedqa.log" 2>&1 & p3=$!
wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"

run_reader 2wikimultihopqa unifiedqa allenai/unifiedqa-v2-t5-large-1363200 0 >"$ROOT/logs/reader_2wiki_unifiedqa.log" 2>&1 & p4=$!
run_reader musique unifiedqa allenai/unifiedqa-v2-t5-large-1363200 1 >"$ROOT/logs/reader_musique_unifiedqa.log" 2>&1 & p5=$!
wait "$p4"; wait "$p5"

"$PYTHON" "$ROOT/oracle_search/02_oracle_action_landscape.py" \
  --input "$ROOT"/multi_reader/phase_a_smoke/*.jsonl \
  --output-dir "$ROOT/oracle_search/phase_a_smoke/results" \
  --bootstrap-samples 5000 --seed 20260722

echo "V16 Phase-A reader smoke and oracle analysis complete."
