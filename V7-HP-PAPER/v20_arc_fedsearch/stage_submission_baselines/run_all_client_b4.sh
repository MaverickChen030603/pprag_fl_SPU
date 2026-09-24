#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/srv/lab/projects/jiankangchen/pprag_fl_SPU/experiments/proberoute_submission_baselines_20260915}"
REPO="${REPO:-/srv/lab/projects/jiankangchen/pprag_fl_SPU}"
PY="${PY:-/srv/lab/projects/jiankangchen/envs/proberoute-py310/bin/python}"
CODE="$REPO/V7-HP-PAPER/v20_arc_fedsearch/stage_submission_baselines"
RUN="${RUN:-$BASE/runs/all_client_b4_r5_posthoc_20260924}"
INPUT="$BASE/inputs/r5_run"
V17="$BASE/inputs/v17"
V16_EVAL="$BASE/inputs/v16_evaluation"
HF_CACHE="${HF_CACHE:-$BASE/huggingface_cache}"
METHOD="b4_all_client_retrieval"

mkdir -p "$RUN/logs" "$RUN/reader_predictions"

if [[ ! -f "$RUN/routes/route_manifest.json" ]]; then
  "$PY" "$CODE/route_all_client_b4.py" \
    --input-root "$INPUT" \
    --output-dir "$RUN/routes" \
    2>&1 | tee "$RUN/logs/route.log"
fi

if [[ ! -f "$RUN/materialized/materialization_manifest.json" ]]; then
  "$PY" "$CODE/materialize_single_route_method.py" \
    --method "$METHOD" \
    --client-budget -1 \
    --input-root "$INPUT" \
    --index-root "$BASE/inputs/indexes" \
    --route-root "$RUN/routes" \
    --output-dir "$RUN/materialized" \
    2>&1 | tee "$RUN/logs/materialize.log"
fi

pids=()
for reader in flan unifiedqa; do
  gpu=0; [[ "$reader" == unifiedqa ]] && gpu=1
  prediction="$RUN/reader_predictions/${reader}_unscored.jsonl"
  if [[ ! -f "${prediction}.completed.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$HF_CACHE" "$PY" "$CODE/run_reader_unscored.py" \
      --reader "$reader" \
      --contexts "$RUN/materialized/contexts_unlabeled.jsonl" \
      --sample-root "$INPUT/protocol" \
      --v16-eval "$V16_EVAL" \
      --cache-dir "$HF_CACHE" \
      --output "$prediction" \
      --device cuda \
      --batch-size 8 \
      --resume \
      >"$RUN/logs/${reader}.log" 2>&1 &
    pids+=("$!")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ ! -f "$RUN/evaluation/evaluation_manifest.json" ]]; then
  "$PY" "$CODE/evaluate_single_route_method.py" \
    --method "$METHOD" \
    --contexts "$RUN/materialized/contexts_unlabeled.jsonl" \
    --prediction-dir "$RUN/reader_predictions" \
    --sample-root "$INPUT/protocol" \
    --label-root "$V17/data/sealed" \
    --assignment-root "$V17/partitions/assignments" \
    --packet-root "$INPUT/retrieval" \
    --v16-eval "$V16_EVAL" \
    --historical-per-query "$INPUT/statistics/per_query_results.csv" \
    --b0-per-query "$BASE/runs/posthoc_r5_20260915/evaluation/per_query_results.csv" \
    --output-dir "$RUN/evaluation" \
    2>&1 | tee "$RUN/logs/evaluate.log"
fi
