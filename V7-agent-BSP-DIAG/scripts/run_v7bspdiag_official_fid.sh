#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v7agentbspdiag}"
UPSTREAM="${UPSTREAM:-$BASE/outputs/$EXPERIMENT_NAME}"
OUTROOT="${OUTROOT:-$BASE/eval_outputs/official_fid_t5}"
RAWDATA="${RAWDATA:-/home/iiserver31/projects/FedE4RAG-main/FedE/select_data_hotpot_train_5000.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
DEVICE="${DEVICE:-cpu}"
LOG_DIR="$BASE/runs/logs"; mkdir -p "$LOG_DIR" "$OUTROOT"
LOG="$LOG_DIR/official_fid_diag_$(date '+%Y%m%d_%H%M%S').log"
bash "$BASE/scripts/check_fid_reader.sh" | tee -a "$LOG"
mapfile -t RUNS < <(find "$UPSTREAM" -name final_artifacts.json -printf '%h
' | sort)
echo "[$(date '+%F %T')] START DIAG official runs=${#RUNS[@]}" | tee -a "$LOG"
idx=0
for RUN in "${RUNS[@]}"; do
  idx=$((idx+1)); REL="${RUN#$UPSTREAM/}"; SUITE="${REL%%/*}"; NAME="$(basename "$RUN")"; OUT="$OUTROOT/$SUITE/$NAME"
  [[ -s "$OUT/official_metrics.json" ]] && { echo "[$idx/${#RUNS[@]}] SKIP $SUITE/$NAME" | tee -a "$LOG"; continue; }
  echo "[$idx/${#RUNS[@]}] RUN $SUITE/$NAME" | tee -a "$LOG"
  "$PYTHON_BIN" "$BASE/run_hotpot_official_eval.py" --run-dir "$RUN" --rawdata-path "$RAWDATA" --output-dir "$OUT" --max-examples "$MAX_EXAMPLES" --support-topk 2 --answer-topk 5 --batch-size 8 --device "$DEVICE" --reader fid --fid-model t5-base --fid-num-beams 3 --fid-max-input-length 768 --fid-max-answer-length 32 --passage-ordering retrieval_score 2>&1 | tee -a "$LOG"
done
