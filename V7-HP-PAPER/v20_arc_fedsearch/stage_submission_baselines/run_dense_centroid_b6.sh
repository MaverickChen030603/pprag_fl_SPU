#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/srv/lab/projects/jiankangchen/pprag_fl_SPU/experiments/proberoute_submission_baselines_20260915}"
REPO="${REPO:-/srv/lab/projects/jiankangchen/pprag_fl_SPU}"
PY="${PY:-/srv/lab/projects/jiankangchen/envs/proberoute-py310/bin/python}"
CODE="$REPO/V7-HP-PAPER/v20_arc_fedsearch/stage_submission_baselines"
RUN="${RUN:-$BASE/runs/dense_centroid_b6_r5_posthoc_20260918}"
INPUT="$BASE/inputs/r5_run"
V17="$BASE/inputs/v17"
V16_EVAL="$BASE/inputs/v16_evaluation"
HF_CACHE="${HF_CACHE:-$BASE/huggingface_cache}"
MODEL="BAAI/bge-base-en-v1.5"
REVISION="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
METHOD="b6_dense_centroid_top3"
B3_RUN="${B3_RUN:-$BASE/runs/ragroute_b3_r5_posthoc_20260916}"

mkdir -p "$RUN/logs" "$RUN/reader_predictions"

declare -A GPU=( [hotpotqa]=0 [2wikimultihopqa]=1 [musique]=2 )
pids=()
for dataset in hotpotqa 2wikimultihopqa musique; do
  route_dir="$RUN/routes/$dataset"
  if [[ ! -f "$route_dir/route_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU[$dataset]}" HF_HOME="$HF_CACHE" "$PY" "$CODE/route_dense_centroid_b6.py" \
      --dataset "$dataset" \
      --centroids "$B3_RUN/centroids/$dataset/source_centroids.npy" \
      --r5-packets "$INPUT/retrieval/${dataset}_probe_packets.jsonl" \
      --r5-inputs "$INPUT/protocol/${dataset}_final_test_inputs_n300.jsonl" \
      --model "$MODEL" \
      --revision "$REVISION" \
      --output-dir "$route_dir" \
      --device cuda \
      >"$RUN/logs/route_${dataset}.log" 2>&1 &
    pids+=("$!")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ ! -f "$RUN/materialized/materialization_manifest.json" ]]; then
  "$PY" "$CODE/materialize_single_route_method.py" \
    --method "$METHOD" \
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
