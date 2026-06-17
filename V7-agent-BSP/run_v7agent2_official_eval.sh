#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BASE/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
UPSTREAM="${UPSTREAM:-$BASE/outputs/pprag_fl_v7agent2}"
OUTROOT="${OUTROOT:-$BASE/outputs/hotpot_official_eval/v7agent2_all}"
RAWDATA="${RAWDATA:-$ROOT/FedE/select_data_hotpot_train_5000.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
DEVICE="${DEVICE:-cpu}"
READER="${READER:-fid}"
FID_MODEL="${FID_MODEL:-t5-base}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SUPPORT_TOPK="${SUPPORT_TOPK:-2}"
ANSWER_TOPK="${ANSWER_TOPK:-5}"
LOG="${LOG:-$ROOT/v7agent2_official_eval.log}"
mkdir -p "$OUTROOT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
mapfile -t RUNS < <(find "$UPSTREAM" -name final_artifacts.json -printf '%h
' | sort)
log "START V7-agent-2 official eval runs=${#RUNS[@]} examples=$MAX_EXAMPLES reader=$READER fid=$FID_MODEL"
idx=0
for RUN in "${RUNS[@]}"; do
  idx=$((idx+1))
  REL="${RUN#$UPSTREAM/}"
  SUITE="${REL%%/*}"
  NAME="$(basename "$RUN")"
  OUT="$OUTROOT/$SUITE/$NAME"
  if [[ -s "$OUT/official_metrics.json" ]]; then log "[$idx/${#RUNS[@]}] SKIP $NAME"; continue; fi
  log "[$idx/${#RUNS[@]}] RUN $NAME"
  "$PYTHON_BIN" "$BASE/run_hotpot_official_eval.py"     --run-dir "$RUN" --rawdata-path "$RAWDATA" --output-dir "$OUT"     --max-examples "$MAX_EXAMPLES" --support-topk "$SUPPORT_TOPK" --answer-topk "$ANSWER_TOPK"     --batch-size "$BATCH_SIZE" --device "$DEVICE" --reader "$READER" --fid-model "$FID_MODEL" 2>&1 | tee -a "$LOG"
done
log "DONE V7-agent-2 official eval"
