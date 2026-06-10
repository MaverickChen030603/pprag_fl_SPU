#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
UPSTREAM="${UPSTREAM:-$ROOT/V7-agent/outputs/pprag_fl_v7_agent/hp1_budget_aligned}"
OUTROOT="${OUTROOT:-$ROOT/V7-agent/outputs/hotpot_official_eval/hp1_budget_aligned}"
RAWDATA="${RAWDATA:-$ROOT/FedE/select_data_hotpot_train_5000.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SUPPORT_TOPK="${SUPPORT_TOPK:-2}"
ANSWER_TOPK="${ANSWER_TOPK:-5}"
LOG="${LOG:-$ROOT/v7_agent_official_eval.log}"
mkdir -p "$OUTROOT"
timestamp(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(timestamp)] $*" | tee -a "$LOG"; }
mapfile -t RUNS < <(find "$UPSTREAM" -name final_artifacts.json -printf '%h\n' | sort)
log "START official Hotpot eval runs=${#RUNS[@]} max_examples=$MAX_EXAMPLES device=$DEVICE out=$OUTROOT"
idx=0
for RUN in "${RUNS[@]}"; do
  idx=$((idx+1))
  NAME="$(basename "$RUN")"
  OUT="$OUTROOT/$NAME"
  if [[ -s "$OUT/official_metrics.json" ]]; then
    log "[$idx/${#RUNS[@]}] SKIP $NAME"
    continue
  fi
  log "[$idx/${#RUNS[@]}] RUN $NAME"
  "$PY" "$ROOT/V7-agent/run_hotpot_official_eval.py" \
    --run-dir "$RUN" \
    --rawdata-path "$RAWDATA" \
    --output-dir "$OUT" \
    --max-examples "$MAX_EXAMPLES" \
    --support-topk "$SUPPORT_TOPK" \
    --answer-topk "$ANSWER_TOPK" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" 2>&1 | tee -a "$LOG"
done
log "DONE official Hotpot eval"
