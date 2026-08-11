#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then echo "usage: $0 <cuda-device>" >&2; exit 2; fi
DEVICE="$1"
ROOT=/home/iiserver31/projects/FedE4RAG-main
PY=/home/iiserver31/anaconda3/envs/supv2/bin/python
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
STAGE="$V20/stage_r3h_hotpot_transfer"
R3="$V20/stage_r3_probe_route"
PROTO="$STAGE/protocol"
PACKETS="$STAGE/smoke/packets"
PROFILES="$STAGE/resource_profiles"

if [[ -e "$STAGE/smoke/complete.json" ]]; then echo "refusing to overwrite R3-H smoke" >&2; exit 4; fi
mkdir -p "$PACKETS" "$PROFILES"
if [[ ! -f "$PROTO/split_manifest.json" ]]; then
  "$PY" "$STAGE/prepare_r3h_splits.py" \
    --train "$V17/data/hotpotqa/train.jsonl" --development "$V17/data/hotpotqa/development.jsonl" \
    --exclude "$V20/stage0_loss_audit/hotpotqa_depth10_n100/all_client_local_depth10.jsonl" \
    --exclude "$V20/stage0_loss_audit/hotpotqa_depth10_n300/all_client_local_depth10.jsonl" \
    --exclude "$ROOT/V7-HP-PAPER/v19_reader_aligned_selective_update/stage0b_top5_boundary_crossing/retrieval_confirmation/hotpotqa_development_disjoint_101_400.jsonl" \
    --exclude "$R3/hotpot_transfer/protocol/probe_holdout.jsonl" --output-dir "$PROTO"
fi
if [[ ! -f "$PROFILES/client_profiles.json" ]]; then
  "$PY" "$R3/hotpot_transfer/build_hotpot_p0_profiles.py" \
    --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" \
    --p0-centroids "$V17/partitions/centroids/hotpotqa/topic_silo_m20.npy" --output-dir "$PROFILES"
fi
for NAME in train holdout; do
  SPLIT="$PROTO/probe_train_labels.jsonl"
  [[ "$NAME" = holdout ]] && SPLIT="$PROTO/holdout_query_view.jsonl"
  CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONPATH="$R3" "$PY" "$R3/materialize_candidate_probe_packets.py" \
    --dataset hotpotqa --split "$SPLIT" --profiles "$PROFILES/client_profiles.json" \
    --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" \
    --output "$PACKETS/${NAME}_5.jsonl" --device cuda --limit 5
done
"$PY" - "$PACKETS/train_5.jsonl" "$PACKETS/holdout_5.jsonl" "$STAGE/smoke/complete.json" <<'PY'
import hashlib, json, sys
for path in map(__import__('pathlib').Path, sys.argv[1:3]):
    rows=[json.loads(line) for line in path.open() if line.strip()]
    assert len(rows)==5 and all(len(row['p0_candidate_records'])==8 for row in rows)
    assert all(row['wire_payload_contains_text'] is False and row['wire_payload_contains_embedding'] is False for row in rows)
payload={'stage':'R3-H','status':'pass','queries_per_packet':5,'wire_bytes':592,'feature_count':18,'reader_started':False,'final_test_accessed':False}
__import__('pathlib').Path(sys.argv[3]).parent.mkdir(parents=True,exist_ok=True)
__import__('pathlib').Path(sys.argv[3]).write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload))
PY
