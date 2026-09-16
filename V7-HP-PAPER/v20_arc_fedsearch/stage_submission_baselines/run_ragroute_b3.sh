#!/usr/bin/env bash
set -euo pipefail

# R5 is historically revealed.  This runner is a reproducible diagnostic, not
# a fresh confirmation experiment; labels remain unread until Reader outputs.
BASE="${BASE:-/srv/lab/projects/jiankangchen/pprag_fl_SPU/experiments/proberoute_submission_baselines_20260915}"
REPO="${REPO:-/srv/lab/projects/jiankangchen/pprag_fl_SPU}"
PY="${PY:-/srv/lab/projects/jiankangchen/envs/proberoute-py310/bin/python}"
CODE="$REPO/V7-HP-PAPER/v20_arc_fedsearch/stage_submission_baselines"
RUN="${RUN:-$BASE/runs/ragroute_b3_r5_posthoc_20260916}"
INPUT="$BASE/inputs/r5_run"
V17="$BASE/inputs/v17"
V16_EVAL="$BASE/inputs/v16_evaluation"
HF_CACHE="${HF_CACHE:-$BASE/huggingface_cache}"
MODEL="BAAI/bge-base-en-v1.5"
REVISION="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
PUBLIC="$BASE/public_training_sources"

mkdir -p "$RUN/logs" "$PUBLIC"

declare -A SOURCE
SOURCE[hotpotqa]="$PUBLIC/hotpot_train_v1.1.json"
SOURCE[2wikimultihopqa]="$PUBLIC/2wikimultihop_train.json"
SOURCE[musique]="$PUBLIC/musique_ans_v1.0_train.jsonl"
for dataset in hotpotqa 2wikimultihopqa musique; do
  [[ -s "${SOURCE[$dataset]}" ]] || { echo "missing verified public source: ${SOURCE[$dataset]}" >&2; exit 2; }
done

# Download once before three GPU workers access the exact pinned BGE revision.
HF_HOME="$HF_CACHE" "$PY" - <<PY
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained("$MODEL", revision="$REVISION")
AutoModel.from_pretrained("$MODEL", revision="$REVISION")
print("BGE pinned revision available")
PY

for dataset in hotpotqa 2wikimultihopqa musique; do
  train_dir="$RUN/training_data/$dataset"
  if [[ ! -f "$train_dir/training_split_manifest.json" ]]; then
    "$PY" "$CODE/prepare_public_router_training_data.py" --dataset "$dataset" --public-source "${SOURCE[$dataset]}" --r5-inputs "$INPUT/protocol" --output-dir "$train_dir" 2>&1 | tee "$RUN/logs/prepare_${dataset}.log"
  fi
done

declare -A GPU=( [hotpotqa]=0 [2wikimultihopqa]=1 [musique]=2 )
pids=()
for dataset in hotpotqa 2wikimultihopqa musique; do
  asset_dir="$RUN/centroids/$dataset"
  if [[ ! -f "$asset_dir/centroid_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU[$dataset]}" HF_HOME="$HF_CACHE" "$PY" "$CODE/build_ragroute_centroids.py" --dataset "$dataset" --index "$BASE/inputs/indexes/$dataset.sqlite" --assignment "$V17/partitions/assignments/$dataset/topic_silo_m20.jsonl" --model "$MODEL" --revision "$REVISION" --output-dir "$asset_dir" --device cuda 2>&1 | tee "$RUN/logs/centroids_${dataset}.log" &
    pids+=("$!")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
for dataset in hotpotqa 2wikimultihopqa musique; do
  route_dir="$RUN/routes/$dataset"
  if [[ ! -f "$route_dir/training_and_route_manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU[$dataset]}" HF_HOME="$HF_CACHE" "$PY" "$CODE/train_ragroute_b3.py" --dataset "$dataset" --train "$RUN/training_data/$dataset/router_train_public.jsonl" --assignment "$V17/partitions/assignments/$dataset/topic_silo_m20.jsonl" --centroids "$RUN/centroids/$dataset/source_centroids.npy" --r5-packets "$INPUT/retrieval/${dataset}_probe_packets.jsonl" --r5-inputs "$INPUT/protocol/${dataset}_final_test_inputs_n300.jsonl" --model "$MODEL" --revision "$REVISION" --v16-eval "$V16_EVAL" --output-dir "$route_dir" --device cuda 2>&1 | tee "$RUN/logs/train_${dataset}.log" &
    pids+=("$!")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ ! -f "$RUN/materialized/materialization_manifest.json" ]]; then
  "$PY" "$CODE/materialize_ragroute_b3.py" --input-root "$INPUT" --index-root "$BASE/inputs/indexes" --route-root "$RUN/routes" --output-dir "$RUN/materialized" 2>&1 | tee "$RUN/logs/materialize.log"
fi

pids=()
for reader in flan unifiedqa; do
  gpu=0; [[ "$reader" == unifiedqa ]] && gpu=1
  prediction="$RUN/reader_predictions/${reader}_unscored.jsonl"
  mkdir -p "$RUN/reader_predictions"
  if [[ ! -f "${prediction}.completed.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$HF_CACHE" "$PY" "$CODE/run_reader_unscored.py" --reader "$reader" --contexts "$RUN/materialized/contexts_unlabeled.jsonl" --sample-root "$INPUT/protocol" --v16-eval "$V16_EVAL" --cache-dir "$HF_CACHE" --output "$prediction" --device cuda --batch-size 8 --resume >"$RUN/logs/${reader}.log" 2>&1 &
    pids+=("$!")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

if [[ ! -f "$RUN/evaluation/evaluation_manifest.json" ]]; then
  "$PY" "$CODE/evaluate_ragroute_b3.py" --contexts "$RUN/materialized/contexts_unlabeled.jsonl" --prediction-dir "$RUN/reader_predictions" --sample-root "$INPUT/protocol" --label-root "$V17/data/sealed" --assignment-root "$V17/partitions/assignments" --packet-root "$INPUT/retrieval" --v16-eval "$V16_EVAL" --historical-per-query "$INPUT/statistics/per_query_results.csv" --b0-per-query "$BASE/runs/posthoc_r5_20260915/evaluation/per_query_results.csv" --output-dir "$RUN/evaluation" 2>&1 | tee "$RUN/logs/evaluate.log"
fi
