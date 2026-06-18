#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v7agentbsp}"
UPSTREAM="${UPSTREAM:-$BASE/outputs/$EXPERIMENT_NAME}"
OUTROOT="${OUTROOT:-$BASE/eval_outputs/official_fid_t5}"
RAWDATA="${RAWDATA:-/home/iiserver31/projects/FedE4RAG-main/FedE/select_data_hotpot_train_5000.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FID_MODEL="${FID_MODEL:-t5-base}"
FID_NUM_BEAMS="${FID_NUM_BEAMS:-3}"
FID_MAX_INPUT_LENGTH="${FID_MAX_INPUT_LENGTH:-768}"
FID_MAX_OUTPUT_LENGTH="${FID_MAX_OUTPUT_LENGTH:-32}"
PASSAGE_ORDERING="${PASSAGE_ORDERING:-retrieval_score}"
LOG_DIR="$BASE/runs/logs"; mkdir -p "$LOG_DIR" "$OUTROOT"
LOG="$LOG_DIR/official_fid_$(date '+%Y%m%d_%H%M%S').log"
bash "$BASE/scripts/check_fid_reader.sh" | tee -a "$LOG"
mapfile -t RUNS < <(find "$UPSTREAM" -name final_artifacts.json -printf '%h
' | sort)
echo "[$(date '+%F %T')] START official FiD/T5 runs=${#RUNS[@]} examples=$MAX_EXAMPLES device=$DEVICE beams=$FID_NUM_BEAMS max_input=$FID_MAX_INPUT_LENGTH ordering=$PASSAGE_ORDERING" | tee -a "$LOG"
idx=0
for RUN in "${RUNS[@]}"; do
  idx=$((idx+1))
  REL="${RUN#$UPSTREAM/}"
  SUITE="${REL%%/*}"
  NAME="$(basename "$RUN")"
  OUT="$OUTROOT/$SUITE/$NAME"
  if [[ -s "$OUT/official_metrics.json" ]]; then echo "[$idx/${#RUNS[@]}] SKIP $SUITE/$NAME" | tee -a "$LOG"; continue; fi
  echo "[$idx/${#RUNS[@]}] RUN $SUITE/$NAME" | tee -a "$LOG"
  "$PYTHON_BIN" "$BASE/run_hotpot_official_eval.py" --run-dir "$RUN" --rawdata-path "$RAWDATA" --output-dir "$OUT"     --max-examples "$MAX_EXAMPLES" --support-topk 2 --answer-topk 5 --batch-size "$BATCH_SIZE"     --device "$DEVICE" --reader fid --fid-model "$FID_MODEL" --fid-num-beams "$FID_NUM_BEAMS"     --fid-max-input-length "$FID_MAX_INPUT_LENGTH" --fid-max-answer-length "$FID_MAX_OUTPUT_LENGTH"     --passage-ordering "$PASSAGE_ORDERING" 2>&1 | tee -a "$LOG"
done
echo "[$(date '+%F %T')] DONE official FiD/T5" | tee -a "$LOG"
