#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/projects/FedE4RAG-main}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-supv2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED_LIST="${SEED_LIST:-0,1,2}"
EMBEDDING_MODEL="${FEDE_EMBEDDING_MODEL:-BAAI/bge-base-en-v1.5}"
RAG_SCRIPT="${RAG_SCRIPT:-main_100_test.py}"
FORCE_RAG="${FORCE_RAG:-0}"

UPSTREAM_BASE="$ROOT_DIR/V3/outputs/pprag_fl_v3"
DOWNSTREAM_BASE="$ROOT_DIR/V3/outputs/rag_eval_all_v3"
REPORT_ROOT="$ROOT_DIR/实验分析报告/V3"

mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

export FEDE_EMBEDDING_MODEL="$EMBEDDING_MODEL"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

suite_report_exists() {
  local suite="$1"
  compgen -G "$REPORT_ROOT/suite_${suite}_*" > /dev/null
}

full_report_exists() {
  local suite="$1"
  compgen -G "$REPORT_ROOT/full_pipeline_${suite}_*" > /dev/null
}

suite_pid() {
  local suite="$1"
  pgrep -f "python V3/run_experiment_suite.py --suite ${suite}" | tail -n 1 || true
}

wait_for_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  while kill -0 "$pid" 2>/dev/null; do
    log "waiting for ${label} (pid=${pid})"
    sleep 120
  done
}

run_suite() {
  local suite="$1"
  local logfile="$ROOT_DIR/${suite}.log"

  if suite_report_exists "$suite"; then
    log "suite ${suite} already completed; skipping rerun"
    return 0
  fi

  local pid
  pid="$(suite_pid "$suite")"
  if [[ -n "$pid" ]]; then
    log "suite ${suite} already running with pid=${pid}; waiting"
    wait_for_pid "$pid" "suite ${suite}"
    return 0
  fi

  log "starting suite ${suite}"
  $PYTHON_BIN V3/run_experiment_suite.py \
    --suite "$suite" \
    --seed-list "$SEED_LIST" \
    --gpu "$GPU_ID" \
    --batch-size "$BATCH_SIZE" \
    > "$logfile" 2>&1
  log "suite ${suite} finished"
}

finalize_suite() {
  local suite="$1"
  local logfile="$ROOT_DIR/${suite}_finalize.log"
  local upstream_root="$UPSTREAM_BASE/$suite"
  local downstream_root="$DOWNSTREAM_BASE/$suite"

  if full_report_exists "$suite"; then
    log "full pipeline report for ${suite} already exists; skipping finalize"
    return 0
  fi

  log "finalizing suite ${suite}"
  if [[ "$FORCE_RAG" == "1" ]]; then
    $PYTHON_BIN V3/finalize_pipeline.py \
      --suite-name "$suite" \
      --upstream-root "$upstream_root" \
      --downstream-root "$downstream_root" \
      --script "$RAG_SCRIPT" \
      --python "$PYTHON_BIN" \
      --force-rag \
      > "$logfile" 2>&1
  else
    $PYTHON_BIN V3/finalize_pipeline.py \
      --suite-name "$suite" \
      --upstream-root "$upstream_root" \
      --downstream-root "$downstream_root" \
      --script "$RAG_SCRIPT" \
      --python "$PYTHON_BIN" \
      > "$logfile" 2>&1
  fi
  log "finalize ${suite} finished"
}

finalize_all() {
  local suite="all_v3"
  local logfile="$ROOT_DIR/${suite}_finalize.log"
  if full_report_exists "$suite"; then
    log "global ${suite} report already exists; skipping"
    return 0
  fi

  log "running global ${suite} finalize"
  if [[ "$FORCE_RAG" == "1" ]]; then
    $PYTHON_BIN V3/finalize_pipeline.py \
      --suite-name "$suite" \
      --upstream-root "$UPSTREAM_BASE" \
      --downstream-root "$DOWNSTREAM_BASE/$suite" \
      --script "$RAG_SCRIPT" \
      --python "$PYTHON_BIN" \
      --force-rag \
      > "$logfile" 2>&1
  else
    $PYTHON_BIN V3/finalize_pipeline.py \
      --suite-name "$suite" \
      --upstream-root "$UPSTREAM_BASE" \
      --downstream-root "$DOWNSTREAM_BASE/$suite" \
      --script "$RAG_SCRIPT" \
      --python "$PYTHON_BIN" \
      > "$logfile" 2>&1
  fi
  log "global ${suite} finalize finished"
}

log "run_v3_all starting"
for suite in v3_main v3_budget v3_heterogeneity v3_ablation_feature v3_ablation_budget v3_explain; do
  run_suite "$suite"
  finalize_suite "$suite"
done
finalize_all
log "run_v3_all completed"
