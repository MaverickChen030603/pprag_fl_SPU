#!/usr/bin/env bash
# Launch both REM-P candidate-recovery runs on the server.
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
STAGE="$V20/stage_remp_evidence_memory"
LOG_DIR="$STAGE/logs"
mkdir -p "$LOG_DIR"

GPU_2WIKI="${REMP_GPU_2WIKI:-0}"
GPU_MUSIQUE="${REMP_GPU_MUSIQUE:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"

nohup bash "$STAGE/run_remp_dataset.sh" 2wikimultihopqa "$GPU_2WIKI" \
  > "$LOG_DIR/remp_2wikimultihopqa_${STAMP}.log" 2>&1 &
PID_2WIKI=$!

nohup bash "$STAGE/run_remp_dataset.sh" musique "$GPU_MUSIQUE" \
  > "$LOG_DIR/remp_musique_${STAMP}.log" 2>&1 &
PID_MUSIQUE=$!

cat > "$LOG_DIR/remp_launch_${STAMP}.json" <<JSON
{
  "stage": "REM-P",
  "timestamp": "$STAMP",
  "root": "$ROOT",
  "runs": [
    {
      "dataset": "2wikimultihopqa",
      "gpu": "$GPU_2WIKI",
      "pid": $PID_2WIKI,
      "log": "$LOG_DIR/remp_2wikimultihopqa_${STAMP}.log"
    },
    {
      "dataset": "musique",
      "gpu": "$GPU_MUSIQUE",
      "pid": $PID_MUSIQUE,
      "log": "$LOG_DIR/remp_musique_${STAMP}.log"
    }
  ],
  "reader_started": false,
  "final_test_accessed": false
}
JSON

cat "$LOG_DIR/remp_launch_${STAMP}.json"

