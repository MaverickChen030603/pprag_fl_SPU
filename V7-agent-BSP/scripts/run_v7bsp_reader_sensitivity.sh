#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD_FILTER="${METHOD_FILTER:-agent-rule-v7-dynamic|agent-pm-bandit-slot|agent-bsp-memory-bandit-strict|agent-bsp-memory-bandit-retrieval|agent-bsp-memory-bandit-reader}"
MAX_EXAMPLES="${MAX_EXAMPLES:-300}"
DEVICE="${DEVICE:-cpu}"
for BEAM in 1 3 5; do
  for LEN in 512 768 1024; do
    for ORDER in retrieval_score agent_priority gold_oracle_debug; do
      OUTROOT="$BASE/eval_outputs/reader_sensitivity/beam${BEAM}_len${LEN}_${ORDER}"
      UPSTREAM="$BASE/outputs/pprag_fl_v7agentbsp"
      mkdir -p "$OUTROOT"
      mapfile -t RUNS < <(find "$UPSTREAM" -name final_artifacts.json -printf '%h
' | grep -E "$METHOD_FILTER" | sort)
      echo "[$(date '+%F %T')] sensitivity beam=$BEAM len=$LEN order=$ORDER runs=${#RUNS[@]}"
      for RUN in "${RUNS[@]}"; do
        REL="${RUN#$UPSTREAM/}"; SUITE="${REL%%/*}"; NAME="$(basename "$RUN")"; OUT="$OUTROOT/$SUITE/$NAME"
        [[ -s "$OUT/official_metrics.json" ]] && continue
        /home/iiserver31/anaconda3/envs/supv2/bin/python "$BASE/run_hotpot_official_eval.py" --run-dir "$RUN" --rawdata-path /home/iiserver31/projects/FedE4RAG-main/FedE/select_data_hotpot_train_5000.json --output-dir "$OUT"           --max-examples "$MAX_EXAMPLES" --support-topk 2 --answer-topk 5 --batch-size 8 --device "$DEVICE" --reader fid --fid-model t5-base           --fid-num-beams "$BEAM" --fid-max-input-length "$LEN" --fid-max-answer-length 32 --passage-ordering "$ORDER"
      done
    done
  done
done
