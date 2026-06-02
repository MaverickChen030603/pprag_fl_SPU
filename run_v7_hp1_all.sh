#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MODE="${1:-full}"
LOG="${LOG:-$ROOT/v7_hp1_all.log}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pprag_fl_v7_hp1}"
ROUNDS="${HP1_ROUNDS:-12}"
CLIENTS="${HP1_CLIENTS:-5}"
EPOCHS="${HP1_EPOCHS:-1}"
BATCH_SIZE="${HP1_BATCH_SIZE:-8}"
LR="${HP1_LR:-1e-5}"
GPU="${HP1_GPU:-0}"
SEED_LIST="${HP1_SEED_LIST:-0,1,2}"
RAWDATA_PATH="${HP1_RAWDATA_PATH:-$ROOT/FedE/select_data_hotpot_train_5000.json}"
RAG_HOTPOT_MAX_EXAMPLES="${HP1_RAG_HOTPOT_MAX_EXAMPLES:-1000}"
RUN_STANDARD_RAG="${RUN_STANDARD_RAG:-0}"

mkdir -p "$ROOT/V7-HP1/outputs" "$ROOT/实验分析报告/V7-HP1"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG"; }

run_suite() {
  local suite="$1"
  log "START suite=$suite rounds=$ROUNDS seeds=$SEED_LIST rawdata=$RAWDATA_PATH"
  "$PYTHON_BIN" "$ROOT/V7-HP1/run_experiment_suite.py" \
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

  local suite_root="$ROOT/V7-HP1/outputs/$EXPERIMENT_NAME/$suite"
  log "SUMMARIZE suite=$suite"
  "$PYTHON_BIN" "$ROOT/V7-HP1/summarize_results.py" \
    --root "$suite_root" \
    --output "$suite_root/summary" 2>&1 | tee -a "$LOG" || true

  log "STRICT_EVAL suite=$suite"
  "$PYTHON_BIN" "$ROOT/V7-HP1/run_hp1_strict_eval.py" \
    --upstream-root "$suite_root" \
    --output-root "$ROOT/V7-HP1/outputs/hp1_strict_eval/$suite" \
    --force 2>&1 | tee -a "$LOG"

  if [[ "$RUN_STANDARD_RAG" == "1" ]]; then
    log "STANDARD_RAG suite=$suite"
    "$PYTHON_BIN" "$ROOT/V7-HP1/run_all_rag_eval.py" \
      --upstream-root "$suite_root" \
      --output-root "$ROOT/V7-HP1/outputs/rag_eval_all_v7_hp1/$suite" \
      --force 2>&1 | tee -a "$LOG"
  fi
  log "DONE suite=$suite"
}

case "$MODE" in
  smoke)
    SUITES=(smoke)
    ROUNDS="${HP1_ROUNDS:-1}"
    SEED_LIST="${HP1_SEED_LIST:-0}"
    ;;
  full)
    SUITES=(hp1_multihop_hard hp1_rare_bridge_tail hp1_budget_aligned hp1_ablation_signal)
    ;;
  hard)
    SUITES=(hp1_multihop_hard)
    ;;
  rare)
    SUITES=(hp1_rare_bridge_tail)
    ;;
  budget)
    SUITES=(hp1_budget_aligned)
    ;;
  ablation)
    SUITES=(hp1_ablation_signal)
    ;;
  *)
    echo "Usage: $0 {smoke|full|hard|rare|budget|ablation}" >&2
    exit 2
    ;;
esac

if [[ ! -s "$RAWDATA_PATH" ]]; then
  log "Hotpot rawdata missing: $RAWDATA_PATH"
  log "Run: $PYTHON_BIN $ROOT/V7-HP1/prepare_hotpot_data.py --split train --max-examples 5000 --output $RAWDATA_PATH"
  exit 3
fi

log "V7-HP1 HotpotQA pipeline mode=$MODE root=$ROOT"
for suite in "${SUITES[@]}"; do
  run_suite "$suite"
done

log "GLOBAL_SUMMARY"
"$PYTHON_BIN" "$ROOT/V7-HP1/summarize_results.py" \
  --root "$ROOT/V7-HP1/outputs/$EXPERIMENT_NAME" \
  --output "$ROOT/V7-HP1/outputs/$EXPERIMENT_NAME/summary" 2>&1 | tee -a "$LOG" || true

"$PYTHON_BIN" "$ROOT/V7-HP1/run_hp1_strict_eval.py" \
  --upstream-root "$ROOT/V7-HP1/outputs/$EXPERIMENT_NAME" \
  --output-root "$ROOT/V7-HP1/outputs/hp1_strict_eval/all_hp1" \
  --force 2>&1 | tee -a "$LOG" || true

log "WRITE_REPORT"
"$PYTHON_BIN" "$ROOT/V7-HP1/write_hp1_analysis.py" 2>&1 | tee -a "$LOG"
log "V7-HP1 pipeline complete mode=$MODE"
