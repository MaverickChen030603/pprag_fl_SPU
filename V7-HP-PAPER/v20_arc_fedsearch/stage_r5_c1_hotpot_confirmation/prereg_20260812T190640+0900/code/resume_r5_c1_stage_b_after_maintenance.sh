#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
STAGE="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r5_c1_hotpot_confirmation/prereg_20260812T190640+0900"
CODE="$STAGE/code"
FORMAL="$STAGE/formal_execution"
R3="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
V16E="$ROOT/V7-HP-PAPER/v16_action_composition/evaluation"
SPLIT="$STAGE/r5_c1_query_manifest_label_free.jsonl"
ROUTES="$FORMAL/retrieval/hotpotqa_inherited_routes.jsonl"
PACKETS="$FORMAL/retrieval/hotpotqa_probe_packets.jsonl"
CONTEXTS="$FORMAL/retrieval/hotpotqa_contexts_unlabeled.jsonl"
PREDICTIONS="$FORMAL/reader_predictions/flan_unscored.jsonl"
COMPLETION="$FORMAL/reader_predictions/flan_unscored.complete.json"
STAMP="$(date +%Y%m%dT%H%M%S%z)"

"$PY" - "$STAGE/HUMAN_EXECUTION_APPROVAL.json" "$STAGE/r5_c1_execution_contract.json" "$SPLIT" "$ROUTES" "$PACKETS" <<'PY'
import hashlib, json, sys
from pathlib import Path

approval, contract, split, routes, packets = map(Path, sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
value = json.loads(approval.read_text())
if value.get("status") != "R5_C1_STAGE_B_APPROVED" or value.get("contract_sha256") != sha(contract):
    raise SystemExit("approval does not bind the frozen contract")
split_rows = [json.loads(line) for line in split.open() if line.strip()]
split_ids = [str(row["query_id"]) for row in split_rows]
if len(split_ids) != 4200 or len(set(split_ids)) != 4200:
    raise SystemExit("frozen split cardinality mismatch")
route_rows = [json.loads(line) for line in routes.open() if line.strip()]
if len(route_rows) != 4200 or {str(row["query_id"]) for row in route_rows} != set(split_ids):
    raise SystemExit("inherited routes incomplete or mismatched")
packet_rows = [json.loads(line) for line in packets.open() if line.strip()] if packets.exists() else []
packet_ids = [str(row["query_id"]) for row in packet_rows]
if len(packet_ids) != len(set(packet_ids)) or not set(packet_ids).issubset(split_ids):
    raise SystemExit("packet partial has duplicate or foreign IDs")
print(json.dumps({"validated_routes": len(route_rows), "validated_partial_packets": len(packet_rows)}))
PY

if [[ ! -f "$PACKETS" ]] || [[ "$(wc -l < "$PACKETS")" -lt 4200 ]]; then
  PYTHONPATH="$R3" CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/r5_c1_materialize_candidate_probe_packets.py" \
    --dataset hotpotqa --split "$SPLIT" \
    --profiles "$R3/hotpot_transfer/resource_profiles/client_profiles.json" \
    --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" \
    --inherited-routes "$ROUTES" --output "$PACKETS" --device cuda --resume \
    >"$FORMAL/logs/probe_packets_resume_${STAMP}.log" 2>&1
fi

"$PY" - "$SPLIT" "$PACKETS" <<'PY'
import json, sys
from pathlib import Path
split, packets = map(Path, sys.argv[1:])
ids = {str(json.loads(line)["query_id"]) for line in split.open() if line.strip()}
rows = [json.loads(line) for line in packets.open() if line.strip()]
keys = [str(row["query_id"]) for row in rows]
if len(rows) != 4200 or len(set(keys)) != 4200 or set(keys) != ids:
    raise SystemExit("packet completion validation failed")
PY

if [[ ! -f "$CONTEXTS" ]]; then
  "$PY" "$CODE/r5_c1_materialize_contexts.py" \
    --split "$SPLIT" --packets "$PACKETS" --routes "$ROUTES" \
    --model "$R3/hotpot_transfer/models/logistic_seed_20260807.pkl" \
    --index "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite" \
    --output "$CONTEXTS" >"$FORMAL/logs/contexts_resume_${STAMP}.log" 2>&1
elif [[ "$(wc -l < "$CONTEXTS")" -ne 8400 ]]; then
  echo "Nonempty incomplete contexts cannot be overwritten or resumed automatically." >&2
  exit 21
fi

if [[ ! -f "$COMPLETION" ]]; then
  reader_resume=()
  [[ -f "$PREDICTIONS" ]] && reader_resume=(--resume)
  CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/r5_c1_reader_unscored.py" \
    --contexts "$CONTEXTS" --split "$SPLIT" \
    --support-checkpoint "$V16E/checkpoints/hotpotqa_support.joblib" \
    --output "$PREDICTIONS" --device cuda --batch-size 4 "${reader_resume[@]}" \
    >"$FORMAL/logs/flan_unscored_resume_${STAMP}.log" 2>&1
fi

"$PY" - "$PREDICTIONS" "$COMPLETION" <<'PY'
import hashlib, json, sys
from pathlib import Path
predictions, complete = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in predictions.open() if line.strip()]
keys = [(str(row["query_id"]), str(row["method"])) for row in rows]
marker = json.loads(complete.read_text())
actual = hashlib.sha256(predictions.read_bytes()).hexdigest()
if len(rows) != 8400 or len(set(keys)) != 8400 or marker.get("output_sha256") != actual:
    raise SystemExit("Stage B completion validation failed")
PY

sha256sum "$PREDICTIONS" >"$FORMAL/checksums/predictions.sha256"
echo "STAGE_B_COMPLETE_UNSCORED_STOP"
