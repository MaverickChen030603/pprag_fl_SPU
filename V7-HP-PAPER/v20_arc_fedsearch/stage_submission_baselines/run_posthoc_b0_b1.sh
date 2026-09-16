#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/srv/lab/projects/jiankangchen/pprag_fl_SPU/experiments/proberoute_submission_baselines_20260915}"
REPO="${REPO:-/srv/lab/projects/jiankangchen/pprag_fl_SPU}"
PY="${PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
CODE="$REPO/V7-HP-PAPER/v20_arc_fedsearch/stage_submission_baselines"
RUN="${RUN:-$BASE/runs/posthoc_r5_20260915}"
INPUT="$BASE/inputs/r5_run"
V16_EVAL="$BASE/inputs/v16_evaluation"
V17="$BASE/inputs/v17"
HF_CACHE="${HF_CACHE:-$BASE/huggingface_cache}"

mkdir -p "$RUN/logs" "$RUN/reader_predictions"

if [[ ! -f "$V17/data/sealed/reconstruction_manifest.json" ]]; then
  "$PY" "$CODE/reconstruct_r5_labels.py" \
    --sample-root "$INPUT/protocol" \
    --hotpot-source /home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v15_robust_context_repair/data/sources/hotpotqa_distractor_train.jsonl \
    --two-wiki-source /home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v15_robust_context_repair/data/sources/2wikimultihopqa_train.json \
    --musique-source /home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v16_action_composition/data/sources/musique_official/raw_data/musique_ans_train_official_format.jsonl \
    --output-root "$V17/data/sealed" \
    2>&1 | tee "$RUN/logs/reconstruct_labels.log"
fi

if [[ ! -f "$RUN/materialized/materialization_manifest.json" ]]; then
  "$PY" "$CODE/materialize_posthoc_baselines.py" \
    --input-root "$INPUT" \
    --index-root "$BASE/inputs/indexes" \
    --output-dir "$RUN/materialized" \
    2>&1 | tee "$RUN/logs/materialize.log"
fi

FLAN_WEIGHT="$HF_CACHE/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/model.safetensors"
UNIFIED_WEIGHT="$HF_CACHE/models--allenai--unifiedqa-v2-t5-large-1363200/snapshots/1d3b8e13b29dbd161494b0b15428378f4713c418/pytorch_model.bin"
if [[ ! -f "$FLAN_WEIGHT" || ! -f "$UNIFIED_WEIGHT" ]]; then
  "$PY" "$CODE/prepare_reader_models.py" \
    --cache-dir "$HF_CACHE" \
    --manifest "$RUN/reader_model_manifest.json" \
    2>&1 | tee "$RUN/logs/prepare_reader_models.log"
fi

CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/run_reader_unscored.py" \
  --reader flan \
  --contexts "$RUN/materialized/contexts_unlabeled.jsonl" \
  --sample-root "$INPUT/protocol" \
  --v16-eval "$V16_EVAL" \
  --cache-dir "$HF_CACHE" \
  --output "$RUN/reader_predictions/flan_unscored.jsonl" \
  --device cuda --batch-size 8 --resume \
  >"$RUN/logs/flan.log" 2>&1 &
FLAN_PID=$!

CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/run_reader_unscored.py" \
  --reader unifiedqa \
  --contexts "$RUN/materialized/contexts_unlabeled.jsonl" \
  --sample-root "$INPUT/protocol" \
  --v16-eval "$V16_EVAL" \
  --cache-dir "$HF_CACHE" \
  --output "$RUN/reader_predictions/unifiedqa_unscored.jsonl" \
  --device cuda --batch-size 8 --resume \
  >"$RUN/logs/unifiedqa.log" 2>&1 &
UNIFIED_PID=$!

printf '%s\n' "$FLAN_PID" >"$RUN/flan.pid"
printf '%s\n' "$UNIFIED_PID" >"$RUN/unifiedqa.pid"
wait "$FLAN_PID"
wait "$UNIFIED_PID"

"$PY" "$CODE/evaluate_posthoc_baselines.py" \
  --contexts "$RUN/materialized/contexts_unlabeled.jsonl" \
  --prediction-dir "$RUN/reader_predictions" \
  --sample-root "$INPUT/protocol" \
  --label-root "$V17/data/sealed" \
  --assignment-root "$V17/partitions/assignments" \
  --packet-root "$INPUT/retrieval" \
  --v16-eval "$V16_EVAL" \
  --historical-per-query "$INPUT/statistics/per_query_results.csv" \
  --output-dir "$RUN/evaluation" \
  2>&1 | tee "$RUN/logs/evaluate.log"
