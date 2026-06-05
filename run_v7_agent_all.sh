#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MODE="${1:-budget}"
LOG="${LOG:-$ROOT/v7_agent_all.log}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v7_agent}"
ROUNDS="${V7_AGENT_ROUNDS:-12}"
CLIENTS="${V7_AGENT_CLIENTS:-5}"
EPOCHS="${V7_AGENT_EPOCHS:-1}"
BATCH_SIZE="${V7_AGENT_BATCH_SIZE:-8}"
LR="${V7_AGENT_LR:-1e-5}"
GPU="${V7_AGENT_GPU:-0}"
SEED_LIST="${V7_AGENT_SEED_LIST:-0,1,2}"
RAWDATA_PATH="${V7_AGENT_RAWDATA_PATH:-$ROOT/FedE/select_data_hotpot_train_5000.json}"
RAG_HOTPOT_MAX_EXAMPLES="${V7_AGENT_RAG_HOTPOT_MAX_EXAMPLES:-1000}"

mkdir -p "$ROOT/V7-agent/outputs" "$ROOT/实验分析报告/V7-agent"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_suite() {
  local suite="$1"
  log "START V7-agent suite=$suite rounds=$ROUNDS seeds=$SEED_LIST"
  "$PYTHON_BIN" "$ROOT/V7-agent/run_experiment_suite.py" \
    --suite "$suite" \
    --experiment-name "$EXPERIMENT_NAME" \
    --rounds "$ROUNDS" \
    --clients "$CLIENTS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --gpu "$GPU" \
    --seed-list "$SEED_LIST" \
    --rawdata-path "$RAWDATA_PATH" \
    --rag-dataset hotpot_qa \
    --rag-hotpot-split validation \
    --rag-hotpot-max-examples "$RAG_HOTPOT_MAX_EXAMPLES" 2>&1 | tee -a "$LOG"

  local suite_root="$ROOT/V7-agent/outputs/$EXPERIMENT_NAME/$suite"
  log "STRICT_EVAL V7-agent suite=$suite"
  "$PYTHON_BIN" "$ROOT/V7-agent/run_hp1_strict_eval.py" \
    --upstream-root "$suite_root" \
    --output-root "$ROOT/V7-agent/outputs/hp1_strict_eval/$suite" \
    --force 2>&1 | tee -a "$LOG"
  log "DONE V7-agent suite=$suite"
}

case "$MODE" in
  smoke)
    SUITES=(smoke); ROUNDS="${V7_AGENT_ROUNDS:-1}"; SEED_LIST="${V7_AGENT_SEED_LIST:-0}" ;;
  budget)
    SUITES=(hp1_budget_aligned) ;;
  hard)
    SUITES=(hp1_multihop_hard) ;;
  rare)
    SUITES=(hp1_rare_bridge_tail) ;;
  full)
    SUITES=(hp1_budget_aligned hp1_multihop_hard hp1_rare_bridge_tail hp1_ablation_signal) ;;
  *) echo "Usage: $0 {smoke|budget|hard|rare|full}" >&2; exit 2 ;;
esac

if [[ ! -s "$RAWDATA_PATH" ]]; then
  log "Hotpot rawdata missing: $RAWDATA_PATH"
  exit 3
fi

log "V7-agent pipeline mode=$MODE root=$ROOT"
for suite in "${SUITES[@]}"; do
  run_suite "$suite"
done

log "GLOBAL STRICT_EVAL V7-agent"
"$PYTHON_BIN" "$ROOT/V7-agent/run_hp1_strict_eval.py" \
  --upstream-root "$ROOT/V7-agent/outputs/$EXPERIMENT_NAME" \
  --output-root "$ROOT/V7-agent/outputs/hp1_strict_eval/all_hp1" \
  --force 2>&1 | tee -a "$LOG" || true

log "WRITE_REPORT V7-agent"
"$PYTHON_BIN" "$ROOT/V7-agent/write_hp1_analysis.py" 2>&1 | tee -a "$LOG" || true
log "V7-agent pipeline complete mode=$MODE"
