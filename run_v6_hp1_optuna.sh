#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_SUPV2_PY="$HOME/anaconda3/envs/supv2/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SUPV2_PY" ]]; then
    PYTHON_BIN="$DEFAULT_SUPV2_PY"
  else
    PYTHON_BIN="python3"
  fi
fi

GPU_ID="${GPU_ID:-0}"
N_TRIALS="${N_TRIALS:-20}"
ROUNDS="${ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-0}"
WAIT_FOR_V6_HP1="${WAIT_FOR_V6_HP1:-1}"
HOTPOT_TRAIN_EXAMPLES="${HOTPOT_TRAIN_EXAMPLES:-5000}"
HOTPOT_EVAL_EXAMPLES="${HOTPOT_EVAL_EXAMPLES:-300}"
HOTPOT_EVAL_SPLIT="${HOTPOT_EVAL_SPLIT:-validation}"
PAYLOAD_PENALTY="${PAYLOAD_PENALTY:-0.25}"
RAWDATA_PATH="${RAWDATA_PATH:-$ROOT_DIR/FedE/select_data_hotpot_train_${HOTPOT_TRAIN_EXAMPLES}.json}"
LOG_DIR="$ROOT_DIR/V6-HP1-OPTUNA/outputs/logs"

mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/v6_hp1_optuna_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[env] ROOT_DIR=$ROOT_DIR"
echo "[env] PYTHON_BIN=$PYTHON_BIN"
echo "[env] GPU_ID=$GPU_ID N_TRIALS=$N_TRIALS ROUNDS=$ROUNDS BATCH_SIZE=$BATCH_SIZE"
echo "[env] HOTPOT_TRAIN_EXAMPLES=$HOTPOT_TRAIN_EXAMPLES HOTPOT_EVAL_EXAMPLES=$HOTPOT_EVAL_EXAMPLES"

if [[ "$WAIT_FOR_V6_HP1" == "1" ]]; then
  echo "[wait] waiting for existing V6-HP1 automation to finish"
  while pgrep -af 'run_v6_hp1_all.sh|python V6-HP1/run_experiment_suite.py|python V6-HP1/finalize_pipeline.py|python V6-HP1/run_all_rag_eval.py' >/dev/null; do
    date '+[wait] %F %T V6-HP1 still running'
    sleep 300
  done
fi

if [[ ! -f "$RAWDATA_PATH" ]]; then
  echo "[prep] build Hotpot upstream data at $RAWDATA_PATH"
  "$PYTHON_BIN" V6-HP1/prepare_hotpot_data.py \
    --split train \
    --max-examples "$HOTPOT_TRAIN_EXAMPLES" \
    --output "$RAWDATA_PATH"
fi

if ! "$PYTHON_BIN" -c 'import optuna' >/dev/null 2>&1; then
  echo "[prep] installing optuna into current Python environment"
  "$PYTHON_BIN" -m pip install -r V6-HP1-OPTUNA/requirements_optuna.txt
fi

echo "[run] V6-HP1 Optuna search"
"$PYTHON_BIN" V6-HP1-OPTUNA/optuna_search.py \
  --python "$PYTHON_BIN" \
  --gpu "$GPU_ID" \
  --seed "$SEED" \
  --n-trials "$N_TRIALS" \
  --rounds "$ROUNDS" \
  --batch-size "$BATCH_SIZE" \
  --rawdata-path "$RAWDATA_PATH" \
  --hotpot-split "$HOTPOT_EVAL_SPLIT" \
  --eval-examples "$HOTPOT_EVAL_EXAMPLES" \
  --payload-penalty "$PAYLOAD_PENALTY"

echo "run_v6_hp1_optuna completed"

