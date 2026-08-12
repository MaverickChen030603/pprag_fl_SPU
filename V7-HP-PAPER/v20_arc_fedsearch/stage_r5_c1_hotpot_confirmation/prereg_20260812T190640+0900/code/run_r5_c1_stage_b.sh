#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
STAGE="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r5_c1_hotpot_confirmation/prereg_20260812T190640+0900"
CODE="$STAGE/code"
FORMAL="$STAGE/formal_execution"
APPROVAL="$STAGE/HUMAN_EXECUTION_APPROVAL.json"
CONTRACT="$STAGE/r5_c1_execution_contract.json"
SPLIT="$STAGE/r5_c1_query_manifest_label_free.jsonl"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
R3="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route"
V16E="$ROOT/V7-HP-PAPER/v16_action_composition/evaluation"

"$PY" - "$APPROVAL" "$CONTRACT" <<'PY'
import hashlib,json,sys
from pathlib import Path
approval,contract=map(Path,sys.argv[1:])
if not approval.is_file(): raise SystemExit("human execution approval missing")
value=json.loads(approval.read_text())
actual=hashlib.sha256(contract.read_bytes()).hexdigest()
if value.get("status") != "R5_C1_STAGE_B_APPROVED" or value.get("contract_sha256") != actual:
    raise SystemExit("approval does not bind the frozen contract")
PY

mkdir -p "$FORMAL/retrieval" "$FORMAL/reader_predictions" "$FORMAL/logs" "$FORMAL/checksums"
if find "$FORMAL" -type f | grep -q .; then
  echo "Formal attempt already has files. Automatic restart is forbidden; follow the frozen recovery policy." >&2
  exit 8
fi

CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/r5_c1_build_inherited_routes.py" \
  --v17 "$V17" --split "$SPLIT" --output "$FORMAL/retrieval/hotpotqa_inherited_routes.jsonl" --device cuda \
  >"$FORMAL/logs/inherited_routes.log" 2>&1

PYTHONPATH="$R3" CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/r5_c1_materialize_candidate_probe_packets.py" \
  --dataset hotpotqa --split "$SPLIT" \
  --profiles "$R3/hotpot_transfer/resource_profiles/client_profiles.json" \
  --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" \
  --inherited-routes "$FORMAL/retrieval/hotpotqa_inherited_routes.jsonl" \
  --output "$FORMAL/retrieval/hotpotqa_probe_packets.jsonl" --device cuda \
  >"$FORMAL/logs/probe_packets.log" 2>&1

"$PY" "$CODE/r5_c1_materialize_contexts.py" \
  --split "$SPLIT" --packets "$FORMAL/retrieval/hotpotqa_probe_packets.jsonl" \
  --routes "$FORMAL/retrieval/hotpotqa_inherited_routes.jsonl" \
  --model "$R3/hotpot_transfer/models/logistic_seed_20260807.pkl" \
  --index "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite" \
  --output "$FORMAL/retrieval/hotpotqa_contexts_unlabeled.jsonl" \
  >"$FORMAL/logs/contexts.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/r5_c1_reader_unscored.py" \
  --contexts "$FORMAL/retrieval/hotpotqa_contexts_unlabeled.jsonl" --split "$SPLIT" \
  --support-checkpoint "$V16E/checkpoints/hotpotqa_support.joblib" \
  --output "$FORMAL/reader_predictions/flan_unscored.jsonl" --device cuda --batch-size 4 \
  >"$FORMAL/logs/flan_unscored.log" 2>&1

sha256sum "$FORMAL/reader_predictions/flan_unscored.jsonl" >"$FORMAL/checksums/predictions.sha256"
echo "STAGE_B_COMPLETE_UNSCORED_STOP"
