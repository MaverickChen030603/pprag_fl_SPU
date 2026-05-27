#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V6_DIR="$ROOT/V6"
V7_DIR="$ROOT/V7"
REPORT_DIR="$ROOT/实验分析报告/V7"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

log "Bootstrapping V7 from V6"
log "ROOT=$ROOT"

cd "$ROOT" || die "Cannot cd to $ROOT"
test -d "$V6_DIR" || die "Missing V6 directory: $V6_DIR"
test -x "$PYTHON_BIN" || die "Python not executable: $PYTHON_BIN"

if [ -d "$V7_DIR" ]; then
  BACKUP="$ROOT/V7.backup.$STAMP"
  log "Existing V7 found. Moving it to $BACKUP"
  mv "$V7_DIR" "$BACKUP"
fi

log "Copying V6 -> V7"
cp -a "$V6_DIR" "$V7_DIR"
mkdir -p "$V7_DIR/outputs" "$REPORT_DIR"

log "Applying conservative V6->V7 textual migration"
find "$V7_DIR" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' -o -name '*.json' \) -print0 |
  while IFS= read -r -d '' file; do
    perl -0pi -e 's/\bV6\b/V7/g; s/\bv6_/v7_/g; s/\bhypernet_v6\b/hypernet_v6/g; s/\badaptive_v6\b/adaptive_v6/g' "$file"
  done

log "Writing V7 agent method registry overlay"
cat > "$V7_DIR/V7_AGENT_METHODS.md" <<'EOF'
# V7 Agent Method Overlay

This V7 directory was bootstrapped from the working V6 pipeline.

Required V7 methods:

- `agent_rule_v7`: rule-based state/memory/downstream-aware block selector.
- `agent_bandit_v7`: contextual bandit block selector.
- `agent_policy_v7`: learned policy selector.
- `agent_llm_planner_v7`: optional high-level planner that only emits strategy mode and weights.

The first executable pass may map `agent_rule_v7` and `agent_bandit_v7` onto the existing V6
selector interfaces while adding V7 logging fields. Do not let any V7 method exceed the same
payload budget used by `v7_budget_aligned`.
EOF

log "Installing V7 experiment matrix"
cat > "$V7_DIR/v7_experiment_matrix.yaml" <<'EOF'
project:
  name: "Agentic Federated RAG"
  version: "V7"
  root: "/home/iiserver31/projects/FedE4RAG-main"
  python: "/home/iiserver31/anaconda3/envs/supv2/bin/python"
  output_root: "V7/outputs"
  report_root: "实验分析报告/V7"
first_pass:
  suites: [v7_main, v7_budget_aligned, v7_hardquery]
  methods: [random, delta_norm, hypernet_v6, adaptive_v6, agent_rule_v7, agent_bandit_v7]
  seeds: [0, 1, 2]
  topk: [3]
full_pass:
  suites: [v7_main, v7_budget_aligned, v7_heterogeneity, v7_hardquery, v7_ablation_signal, v7_ablation_agent_level, v7_cost_efficiency, v7_explain]
expected_budget:
  first_pass_upstream_runs: 54
  first_pass_downstream_runs: 54
  full_upstream_runs: 148
  full_downstream_runs: 148
EOF

log "Creating V7 run wrapper"
cat > "$ROOT/run_v7_all.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
MODE="${1:-first_pass}"
LOG="$ROOT/v7_all.log"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

run_suite() {
  local suite="$1"
  log "Starting suite: $suite"
  "$PYTHON_BIN" "$ROOT/V7/run_experiment_suite.py" --suite "$suite" 2>&1 | tee -a "$LOG"
  log "Finished upstream suite: $suite"

  if [ -f "$ROOT/V7/run_all_rag_eval.py" ]; then
    log "Starting downstream RAG eval: $suite"
    "$PYTHON_BIN" "$ROOT/V7/run_all_rag_eval.py" --suite "$suite" 2>&1 | tee -a "$LOG" || {
      log "Downstream RAG eval failed for $suite"
      return 1
    }
    log "Finished downstream RAG eval: $suite"
  fi

  if [ -f "$ROOT/V7/finalize_pipeline.py" ]; then
    log "Finalizing suite: $suite"
    "$PYTHON_BIN" "$ROOT/V7/finalize_pipeline.py" --suite "$suite" 2>&1 | tee -a "$LOG" || true
  fi
}

cd "$ROOT"
touch "$LOG"
log "V7 automation started with mode=$MODE"

if [ "$MODE" = "first_pass" ]; then
  SUITES=(v7_main v7_budget_aligned v7_hardquery)
else
  SUITES=(v7_main v7_budget_aligned v7_heterogeneity v7_hardquery v7_ablation_signal v7_ablation_agent_level v7_cost_efficiency v7_explain)
fi

for suite in "${SUITES[@]}"; do
  run_suite "$suite"
done

if [ -f "$ROOT/V7/finalize_pipeline.py" ]; then
  log "Running all_v7 finalize"
  "$PYTHON_BIN" "$ROOT/V7/finalize_pipeline.py" --suite all_v7 2>&1 | tee -a "$LOG" || true
fi

log "V7 automation completed"
EOF
chmod +x "$ROOT/run_v7_all.sh"

log "Creating V7 status checker"
cat > "$ROOT/check_v7_status.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
cd "$ROOT"

echo "== processes =="
pgrep -af 'run_v7_all.sh|python V7/run_experiment_suite.py|python V7/finalize_pipeline.py|python V7/run_all_rag_eval.py' || true

echo
echo "== run counts =="
printf 'run_metadata.json: '
find V7/outputs -name run_metadata.json 2>/dev/null | wc -l
printf 'rag_eval_stdout.log: '
find V7/outputs -name rag_eval_stdout.log 2>/dev/null | wc -l

echo
echo "== V7 reports =="
find 实验分析报告/V7 -maxdepth 2 -type d 2>/dev/null | sort | tail -30 || true

echo
echo "== latest log =="
tail -n 80 v7_all.log 2>/dev/null || true
EOF
chmod +x "$ROOT/check_v7_status.sh"

log "Syntax checking copied Python files"
find "$V7_DIR" -name '*.py' -print0 | xargs -0 -r "$PYTHON_BIN" -m py_compile

log "Bootstrap complete"
log "Next: nohup $ROOT/run_v7_all.sh first_pass > $ROOT/v7_nohup.log 2>&1 &"
