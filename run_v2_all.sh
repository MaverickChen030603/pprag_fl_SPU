#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/iiserver31/FedE4RAG_V2}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-supv2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SEED_LIST="${SEED_LIST:-0,1,2}"
EMBEDDING_MODEL="${FEDE_EMBEDDING_MODEL:-BAAI/bge-base-en-v1.5}"

UPSTREAM_BASE="$ROOT_DIR/SUP_v3/outputs/pprag_fl_sup_v3"
DOWNSTREAM_BASE="$ROOT_DIR/SUP_v3/outputs/rag_eval_all_v2"
REPORT_ROOT="$ROOT_DIR/实验分析报告/V2"

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
  pgrep -f "python SUP_v3/run_experiment_suite.py --suite ${suite}" | tail -n 1 || true
}

finalize_pid() {
  local suite="$1"
  pgrep -f "python SUP_v3/finalize_pipeline.py --suite-name ${suite}" | tail -n 1 || true
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
  local logfile="$ROOT_DIR/v2_${suite}.log"
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
  $PYTHON_BIN SUP_v3/run_experiment_suite.py --suite "$suite" --seed-list "$SEED_LIST" --gpu "$GPU_ID" > "$logfile" 2>&1
  log "suite ${suite} finished"
}

finalize_suite() {
  local suite="$1"
  local logfile="$ROOT_DIR/v2_${suite}_finalize.log"
  local upstream_root="$UPSTREAM_BASE/$suite"
  local downstream_root="$DOWNSTREAM_BASE/$suite"

  if full_report_exists "$suite"; then
    log "full pipeline report for ${suite} already exists; skipping finalize"
    return 0
  fi

  local pid
  pid="$(finalize_pid "$suite")"
  if [[ -n "$pid" ]]; then
    log "finalize ${suite} already running with pid=${pid}; waiting"
    wait_for_pid "$pid" "finalize ${suite}"
    return 0
  fi

  log "finalizing suite ${suite}"
  $PYTHON_BIN SUP_v3/finalize_pipeline.py \
    --suite-name "$suite" \
    --upstream-root "$upstream_root" \
    --downstream-root "$downstream_root" \
    > "$logfile" 2>&1
  log "finalize ${suite} finished"
}

run_encryption_stage() {
  local suite="encryption"
  local upstream_root="$UPSTREAM_BASE/$suite"
  local task_name="num5_dir_a03_imb00_ts0_sup_v3"
  local run_dir="$upstream_root/$task_name/hypernet_k3_w1_s0_enc1"
  local logfile="$ROOT_DIR/v2_encryption.log"
  if [[ -d "$run_dir" ]]; then
    log "encryption run already exists; skipping rerun"
    return 0
  fi

  log "starting encryption estimate run"
  $PYTHON_BIN SUP_v3/run_upstream.py \
    --strategy hypernet \
    --topk 3 \
    --warmup 1 \
    --rounds 25 \
    --gpu "$GPU_ID" \
    --seed 0 \
    --suite-tag "$suite" \
    --partitioner DirichletPartitioner \
    --dir-alpha 0.3 \
    --estimate-encryption \
    > "$logfile" 2>&1
  log "encryption estimate run finished"
}

finalize_all() {
  local logfile="$ROOT_DIR/v2_all_v2_finalize.log"
  if full_report_exists "all_v2"; then
    log "global V2 full pipeline report already exists; skipping final finalize"
    return 0
  fi

  log "running global all_v2 finalize"
  $PYTHON_BIN SUP_v3/run_all_rag_eval.py \
    --upstream-root "$UPSTREAM_BASE" \
    --output-root "$DOWNSTREAM_BASE/all_v2" \
    --script main_100_test.py \
    --python "$PYTHON_BIN" \
    --force \
    >> "$logfile" 2>&1
  $PYTHON_BIN SUP_v3/finalize_pipeline.py \
    --suite-name all_v2 \
    --upstream-root "$UPSTREAM_BASE" \
    --downstream-root "$DOWNSTREAM_BASE/all_v2" \
    >> "$logfile" 2>&1
  log "global all_v2 finalize finished"
}

log "run_v2_all starting"
run_suite "stability"
finalize_suite "stability"

run_suite "heterogeneity"
finalize_suite "heterogeneity"

run_suite "budget"
finalize_suite "budget"

run_suite "topk"
finalize_suite "topk"

run_suite "warmup"
finalize_suite "warmup"

run_encryption_stage
finalize_all
log "run_v2_all completed"
