#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$BASE/实验分析报告/V7-agent-2/run_logs"
mkdir -p "$LOG_DIR"
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
ts(){ date '+%Y%m%d_%H%M%S'; }
echo "=============================="
echo " V7-agent-2 实验套件启动 $(ts)"
echo "=============================="
"$PYTHON_BIN" "$BASE/check_deps.py"
bash "$BASE/run_v7agent2_ablation.sh" 2>&1 | tee "$LOG_DIR/ablation_$(ts).log"
bash "$BASE/run_v7agent2_bandit.sh" 2>&1 | tee "$LOG_DIR/bandit_$(ts).log"
"$PYTHON_BIN" "$BASE/run_experiment_suite.py" --suite v7agent2_dynamic --experiment-name "${EXPERIMENT_NAME:-pprag_fl_v7agent2}" --rounds "${ROUNDS:-12}" --clients 5 --epochs 1 --batch-size "${BATCH_SIZE:-8}" --gpu "${GPU:-0}" --seed-list "${SEED_LIST:-0,1,2,3,4}" --rawdata-path "${RAWDATA_PATH:-$BASE/../FedE/select_data_hotpot_train_5000.json}" --rag-dataset hotpot_qa --rag-hotpot-split validation --rag-hotpot-max-examples 1000 2>&1 | tee "$LOG_DIR/dynamic_$(ts).log"
bash "$BASE/run_v7agent2_official_eval.sh" 2>&1 | tee "$LOG_DIR/official_eval_$(ts).log"
"$PYTHON_BIN" "$BASE/analyze_ablation.py" 2>&1 | tee "$LOG_DIR/analyze_ablation_$(ts).log"
"$PYTHON_BIN" "$BASE/analyze_official_eval.py" 2>&1 | tee "$LOG_DIR/analyze_official_$(ts).log"
"$PYTHON_BIN" "$BASE/generate_v7agent2_report.py"
echo "V7-agent-2 全部实验完成 $(ts)"
