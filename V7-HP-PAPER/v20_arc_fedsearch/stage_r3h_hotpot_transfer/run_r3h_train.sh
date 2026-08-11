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
PACKETS="$STAGE/train/packets"
MODELS="$STAGE/train/models"
PROFILES="$STAGE/resource_profiles"

[[ -f "$STAGE/smoke/complete.json" ]] || { echo "R3-H smoke has not passed" >&2; exit 3; }
[[ -f "$PROTO/split_manifest.json" && -f "$PROFILES/client_profiles.json" ]] || { echo "R3-H protocol not sealed" >&2; exit 3; }
if [[ -e "$MODELS/model_results.csv" ]]; then echo "refusing to overwrite R3-H training" >&2; exit 4; fi
mkdir -p "$PACKETS" "$MODELS"
CUDA_VISIBLE_DEVICES="$DEVICE" PYTHONPATH="$R3" "$PY" "$R3/materialize_candidate_probe_packets.py" \
  --dataset hotpotqa --split "$PROTO/probe_train_labels.jsonl" --profiles "$PROFILES/client_profiles.json" \
  --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" --output "$PACKETS/probe_train.jsonl" --device cuda
PYTHONPATH="$R3" "$PY" "$R3/train_evaluate_logistic_ranker.py" --mode train --dataset hotpotqa \
  --split "$PROTO/probe_train_labels.jsonl" --packets "$PACKETS/probe_train.jsonl" \
  --assignment "$V17/partitions/assignments/hotpotqa/topic_silo_m20.jsonl" --output-dir "$MODELS"
"$PY" - "$MODELS" "$R3/run_compact_payload_audit.py" <<'PY'
import hashlib, json, pickle, sys
from pathlib import Path
models=Path(sys.argv[1]); schema=Path(sys.argv[2])
payload={'feature_schema_sha256':hashlib.sha256(schema.read_bytes()).hexdigest(),'models':{},'reader_started':False,'final_test_accessed':False}
for path in sorted(models.glob('logistic_seed_*.pkl')):
    with path.open('rb') as handle: value=pickle.load(handle)
    payload['models'][path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seed':value['seed'],'coefficient':value['model'].coef_.tolist(),'intercept':value['model'].intercept_.tolist(),'feature_names':value['feature_names']}
(models/'frozen_model_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
PY
