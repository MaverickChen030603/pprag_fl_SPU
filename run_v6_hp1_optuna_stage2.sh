#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/anaconda3/envs/supv2/bin/python}"
GPU_ID="${GPU_ID:-0}"
N_TRIALS="${N_TRIALS:-24}"
ROUNDS="${ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_SUBSET_TYPE="${EVAL_SUBSET_TYPE:-hard_only}"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-1000}"
PAYLOAD_PENALTY_MODE="${PAYLOAD_PENALTY_MODE:-search}"
HARD_QUERY_SUBSET="${HARD_QUERY_SUBSET:-$ROOT_DIR/V6-HP1/data/hotpot_hard_query_subset.json}"

cd "$ROOT_DIR"

if [[ "$EVAL_SUBSET_TYPE" == "hard_only" && ! -f "$HARD_QUERY_SUBSET" ]]; then
  echo "[error] hard query subset not found: $HARD_QUERY_SUBSET" >&2
  echo "[hint] build it with V6-HP1/build_hotpot_hard_subset.py before running hard_only search." >&2
  exit 2
fi

"$PYTHON_BIN" V6-HP1-OPTUNA/v6_hp1_optuna.py \
  --python "$PYTHON_BIN" \
  --gpu "$GPU_ID" \
  --n-trials "$N_TRIALS" \
  --rounds "$ROUNDS" \
  --batch-size "$BATCH_SIZE" \
  --eval-subset-type "$EVAL_SUBSET_TYPE" \
  --eval-num-examples "$EVAL_NUM_EXAMPLES" \
  --hard-query-subset "$HARD_QUERY_SUBSET" \
  --payload-penalty-mode "$PAYLOAD_PENALTY_MODE"
