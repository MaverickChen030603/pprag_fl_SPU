#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
STAGE="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r5_c1_hotpot_confirmation/prereg_20260812T190640+0900"
FORMAL="$STAGE/formal_execution"
CONTRACT="$STAGE/r5_c1_execution_contract.json"
APPROVAL="$STAGE/HUMAN_EXECUTION_APPROVAL.json"
PREDICTIONS="$FORMAL/reader_predictions/flan_unscored.jsonl"
COMPLETION="$FORMAL/reader_predictions/flan_unscored.complete.json"
CONTEXTS="$FORMAL/retrieval/hotpotqa_contexts_unlabeled.jsonl"
SPLIT="$STAGE/r5_c1_query_manifest_label_free.jsonl"
GOLD="$ROOT/V7-HP-PAPER/v15_robust_context_repair/data/sources/hotpotqa_distractor_train.jsonl"
EVAL_COMMON="$ROOT/V7-HP-PAPER/v16_action_composition/evaluation/eval_common.py"
OUTPUT="$FORMAL/evaluation"

"$PY" - "$APPROVAL" "$CONTRACT" <<'PY'
import hashlib, json, sys
from pathlib import Path

approval, contract = map(Path, sys.argv[1:])
if not approval.is_file():
    raise SystemExit("human execution approval missing")
value = json.loads(approval.read_text())
actual = hashlib.sha256(contract.read_bytes()).hexdigest()
if value.get("status") != "R5_C1_STAGE_B_APPROVED" or value.get("contract_sha256") != actual:
    raise SystemExit("approval does not bind the frozen contract")
PY

if [[ ! -f "$COMPLETION" ]]; then
  echo "Stage B completion marker missing; sealed evaluation forbidden." >&2
  exit 9
fi
mkdir -p "$OUTPUT"
if find "$OUTPUT" -type f | grep -q .; then
  echo "Evaluation output already exists; overwrite and rerun are forbidden." >&2
  exit 10
fi

"$PY" "$STAGE/code/r5_c1_sealed_evaluator.py" \
  --predictions "$PREDICTIONS" --completion "$COMPLETION" \
  --contexts "$CONTEXTS" --split "$SPLIT" --gold "$GOLD" \
  --eval-common "$EVAL_COMMON" --output-dir "$OUTPUT"

echo "R5_C1_STAGE_C_COMPLETE"
