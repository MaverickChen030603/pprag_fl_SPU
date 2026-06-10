#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BASE/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
SEED_LIST="${SEED_LIST:-0,1,2}"
ROUNDS="${ROUNDS:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU="${GPU:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v7agent2}"
RAWDATA_PATH="${RAWDATA_PATH:-$ROOT/FedE/select_data_hotpot_train_5000.json}"
LOG="${LOG:-$ROOT/v7agent2_bandit.log}"
mkdir -p "$BASE/outputs" "$ROOT/实验分析报告/V7-agent-2"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "START V7-agent-2 bandit search seeds=$SEED_LIST rounds=$ROUNDS"
"$PYTHON_BIN" "$BASE/run_experiment_suite.py"   --suite v7agent2_bandit   --experiment-name "$EXPERIMENT_NAME"   --rounds "$ROUNDS"   --clients 5   --epochs 1   --batch-size "$BATCH_SIZE"   --gpu "$GPU"   --seed-list "$SEED_LIST"   --rawdata-path "$RAWDATA_PATH"   --rag-dataset hotpot_qa   --rag-hotpot-split validation   --rag-hotpot-max-examples 1000 2>&1 | tee -a "$LOG"
SUITE_ROOT="$BASE/outputs/$EXPERIMENT_NAME/v7agent2_bandit"
"$PYTHON_BIN" "$BASE/run_hp1_strict_eval.py" --upstream-root "$SUITE_ROOT" --output-root "$BASE/outputs/hp1_strict_eval/v7agent2_bandit" --force 2>&1 | tee -a "$LOG"
log "DONE V7-agent-2 bandit search"
