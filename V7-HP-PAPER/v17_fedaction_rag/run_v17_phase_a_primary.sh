#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MAX_QUERIES="${MAX_QUERIES:-20}"
PHASE_TAG="${PHASE_TAG:-phase_a_smoke20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOCAL_SPARSE_CANDIDATES="${LOCAL_SPARSE_CANDIDATES:-20}"
V15="$ROOT/../v15_robust_context_repair"
V16="$ROOT/../v16_action_composition"
PHASE="$ROOT/oracle/$PHASE_TAG"
export PYTHONPATH="$ROOT:$V16"

if [[ ! -f "$ROOT/partitions/query_origin_manifest.json" ]]; then
  echo "Phase-A setup is incomplete: query_origin_manifest.json is missing" >&2
  exit 2
fi
mkdir -p "$ROOT/logs" "$PHASE/pools" "$PHASE/contexts" "$PHASE/reader_outputs" "$PHASE/results"

index_for() {
  case "$1" in
    hotpotqa) echo "$V15/retrieval/indexes/hotpotqa.sqlite" ;;
    2wikimultihopqa) echo "$V15/retrieval/indexes/2wikimultihopqa.sqlite" ;;
    musique) echo "$V16/retrieval/indexes/musique.sqlite" ;;
  esac
}

generate_pool() {
  local dataset="$1" condition="$2" gpu="$3" partition budget
  local split="$ROOT/data/$dataset/development.jsonl"
  local index output
  partition="$condition"
  budget=3
  if [[ "$condition" == "topic_silo_bc2" ]]; then
    partition="topic_silo"
    budget=2
  fi
  index="$(index_for "$dataset")"
  output="$PHASE/pools/${dataset}_${condition}.jsonl"
  if [[ -s "$output" && -s "${output%.jsonl}.manifest.json" ]]; then
    echo "reuse pool $dataset/$partition"
    return
  fi
  if [[ "$partition" == "centralized" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/retrieval/01_generate_federated_pools.py" \
      --dataset "$dataset" --split "$split" --index "$index" --output "$output" \
      --partition centralized --centralized --pool-size 10 --sparse-candidates 5000 \
      --device cuda --max-queries "$MAX_QUERIES"
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/retrieval/01_generate_federated_pools.py" \
      --dataset "$dataset" --split "$split" --index "$index" --output "$output" \
      --partition "$partition" \
      --assignment "$ROOT/partitions/assignments/$dataset/${partition}_m20.jsonl" \
      --centroids "$ROOT/partitions/centroids/$dataset/${partition}_m20.npy" \
      --origins "$ROOT/partitions/client_query_distribution.csv" \
      --client-budget "$budget" --local-k 5 --pool-size 10 \
      --local-sparse-candidates "$LOCAL_SPARSE_CANDIDATES" \
      --device cuda --max-queries "$MAX_QUERIES"
  fi
}

for dataset in hotpotqa 2wikimultihopqa musique; do
  generate_pool "$dataset" centralized 0 >"$ROOT/logs/${PHASE_TAG}_${dataset}_centralized_pool.log" 2>&1 & p0=$!
  generate_pool "$dataset" topic_silo 1 >"$ROOT/logs/${PHASE_TAG}_${dataset}_topic_pool.log" 2>&1 & p1=$!
  generate_pool "$dataset" entity_community 2 >"$ROOT/logs/${PHASE_TAG}_${dataset}_entity_pool.log" 2>&1 & p2=$!
  generate_pool "$dataset" random_control 3 >"$ROOT/logs/${PHASE_TAG}_${dataset}_random_pool.log" 2>&1 & p3=$!
  status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  wait "$p2" || status=1
  wait "$p3" || status=1
  if [[ "$status" -ne 0 ]]; then
    echo "Pool generation failed for $dataset" >&2
    exit "$status"
  fi
  generate_pool "$dataset" topic_silo_bc2 0 \
    >"$ROOT/logs/${PHASE_TAG}_${dataset}_topic_bc2_pool.log" 2>&1
done

"$PYTHON" "$ROOT/oracle/06_audit_routing_metrics.py" \
  --pool-dir "$PHASE/pools" --data-root "$ROOT/data" \
  --dispersion "$ROOT/oracle/evidence_dispersion.csv" \
  --output "$PHASE/results/routing_metrics_per_query.csv" \
  --summary "$PHASE/results/routing_metrics_summary.csv"

generate_contexts() {
  local dataset="$1" condition="$2"
  local output="$PHASE/contexts/${dataset}_${condition}.jsonl"
  if [[ -s "$output" ]]; then
    echo "reuse contexts $dataset/$condition"
    return
  fi
  "$PYTHON" "$ROOT/oracle/02_generate_federated_contexts.py" \
    --pool "$PHASE/pools/${dataset}_${condition}.jsonl" \
    --v16-generator "$V16/oracle_search/01_generate_oracle_trajectories.py" \
    --output "$output" --max-queries "$MAX_QUERIES"
}

context_pids=()
for dataset in hotpotqa 2wikimultihopqa musique; do
  for condition in centralized topic_silo topic_silo_bc2 entity_community random_control; do
    generate_contexts "$dataset" "$condition" \
      >"$ROOT/logs/${PHASE_TAG}_${dataset}_${condition}_contexts.log" 2>&1 &
    context_pids+=("$!")
  done
done
for pid in "${context_pids[@]}"; do wait "$pid"; done

jobs=()
for dataset in hotpotqa 2wikimultihopqa musique; do
  for condition in centralized topic_silo topic_silo_bc2 entity_community random_control; do
    for reader in flan unifiedqa; do
      jobs+=("$dataset|$condition|$reader")
    done
  done
done

reader_worker() {
  local gpu="$1" offset="$2" index dataset condition reader model output
  for ((index=offset; index<${#jobs[@]}; index+=4)); do
    IFS='|' read -r dataset condition reader <<<"${jobs[$index]}"
    if [[ "$reader" == "flan" ]]; then
      model="google/flan-t5-large"
    else
      model="allenai/unifiedqa-v2-t5-large-1363200"
    fi
    output="$PHASE/reader_outputs/${dataset}_${condition}_${reader}.jsonl"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$V16/multi_reader/01_label_oracle_contexts.py" \
      --reader "$reader" --model "$model" --dataset "$dataset" \
      --split "$ROOT/data/$dataset/development.jsonl" \
      --pool "$PHASE/pools/${dataset}_${condition}.jsonl" \
      --contexts "$PHASE/contexts/${dataset}_${condition}.jsonl" \
      --support-predictor "$V16/evaluation/checkpoints/${dataset}_support.joblib" \
      --output "$output" --device cuda --batch-size "$BATCH_SIZE" \
      --max-contexts-per-query 0 --resume
  done
}

reader_worker 0 0 >"$ROOT/logs/${PHASE_TAG}_reader_gpu0.log" 2>&1 & r0=$!
reader_worker 1 1 >"$ROOT/logs/${PHASE_TAG}_reader_gpu1.log" 2>&1 & r1=$!
reader_worker 2 2 >"$ROOT/logs/${PHASE_TAG}_reader_gpu2.log" 2>&1 & r2=$!
reader_worker 3 3 >"$ROOT/logs/${PHASE_TAG}_reader_gpu3.log" 2>&1 & r3=$!
status=0
wait "$r0" || status=1
wait "$r1" || status=1
wait "$r2" || status=1
wait "$r3" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "At least one reader worker failed" >&2
  exit "$status"
fi

analysis_pids=()
for spec in "${jobs[@]}"; do
  IFS='|' read -r dataset condition reader <<<"$spec"
  "$PYTHON" "$ROOT/oracle/03_analyze_federated_oracle.py" \
    --outcomes "$PHASE/reader_outputs/${dataset}_${condition}_${reader}.jsonl" \
    --contexts "$PHASE/contexts/${dataset}_${condition}.jsonl" \
    --pool "$PHASE/pools/${dataset}_${condition}.jsonl" \
    --dispersion "$ROOT/oracle/evidence_dispersion.csv" \
    --output "$PHASE/results/${dataset}_${condition}_${reader}_per_query.csv" \
    --summary "$PHASE/results/${dataset}_${condition}_${reader}_summary.json" \
    --bootstrap 5000 --seed 20260723 \
    >"$ROOT/logs/${PHASE_TAG}_${dataset}_${condition}_${reader}_analysis.log" 2>&1 &
  analysis_pids+=("$!")
done
for pid in "${analysis_pids[@]}"; do wait "$pid"; done

"$PYTHON" "$ROOT/oracle/04_aggregate_phase_a.py" \
  --input-dir "$PHASE/results" --output-dir "$PHASE/results" \
  --bootstrap 5000 --seed 20260723 --minimum-queries 100
"$PYTHON" "$ROOT/oracle/05_plot_opportunity_flow.py" \
  --input "$PHASE/results/federated_opportunity_gap.csv" \
  --output "$PHASE/results/federated_opportunity_flow.pdf"
"$PYTHON" "$ROOT/protocol/03_no_leak_audit.py"
echo "V17 Phase-A primary run complete: $PHASE_TAG, N=$MAX_QUERIES."
