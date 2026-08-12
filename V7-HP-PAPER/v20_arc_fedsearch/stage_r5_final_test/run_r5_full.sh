#!/usr/bin/env bash
# One-shot R5. No metrics are computed before both reader prediction files close.
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
STAGE="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r5_final_test"
RUN="$STAGE/run_20260812"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V16E="$ROOT/V7-HP-PAPER/v16_action_composition/evaluation"

if [[ -e "$RUN/reports/r5_final_decision.json" ]]; then
  echo "R5 already evaluated; refusing rerun" >&2; exit 4
fi
test -f "$RUN/protocol/no_leak_audit.json"
"$PY" "$STAGE/freeze_r5_artifacts.py" --root "$ROOT" --run "$RUN"
cat >"$RUN/cost/r5_cost_contract.json" <<'JSON'
{
  "status": "frozen_before_retrieval",
  "methods": ["M0", "M1", "M2", "M3"],
  "probe_route": {
    "candidate_clients": 8,
    "probe_features_per_client": 18,
    "probe_dtype": "float32",
    "wire_bytes_per_query": 592,
    "selected_clients": 3,
    "local_depth": 10,
    "documents_per_client": 5,
    "transmitted_documents": 15,
    "global_pool": 10,
    "reader_documents": 5
  },
  "reader_contract": {
    "prompt": "V17 frozen prompt",
    "max_context_characters": 4000,
    "max_input_tokens": 1024,
    "decoding": "greedy",
    "max_new_tokens": 32
  }
}
JSON

# Phase 1a: packet materialization. No label path is passed to any process.
"$PY" "$STAGE/build_hotpot_final_origins.py" --v17 "$V17" --sample "$RUN/protocol/hotpotqa_final_test_inputs_n300.jsonl" --output "$RUN/retrieval/hotpotqa_inherited_routes.jsonl" --device cuda:0 >"$RUN/logs/hotpot_origin_replay.log" 2>&1 & p0=$!
PYTHONPATH="$V20/stage_r3_probe_route" CUDA_VISIBLE_DEVICES=1 "$PY" "$V20/stage_r3_probe_route/materialize_candidate_probe_packets.py" --dataset 2wikimultihopqa --split "$RUN/protocol/2wikimultihopqa_final_test_inputs_n300.jsonl" --profiles "$V20/stage_r2_mars_route/2wikimultihopqa/resource_profiles/client_profiles.json" --local-index-root "$V17/retrieval/local_indexes/2wikimultihopqa/topic_silo" --output "$RUN/retrieval/2wikimultihopqa_probe_packets.jsonl" --device cuda --resume >"$RUN/logs/2wiki_packets.log" 2>&1 & p1=$!
PYTHONPATH="$V20/stage_r3_probe_route" CUDA_VISIBLE_DEVICES=2 "$PY" "$V20/stage_r3_probe_route/materialize_candidate_probe_packets.py" --dataset musique --split "$RUN/protocol/musique_final_test_inputs_n300.jsonl" --profiles "$V20/stage_r2_mars_route/musique/resource_profiles/client_profiles.json" --local-index-root "$V17/retrieval/local_indexes/musique/topic_silo" --output "$RUN/retrieval/musique_probe_packets.jsonl" --device cuda --resume >"$RUN/logs/musique_packets.log" 2>&1 & p2=$!
wait "$p0"; wait "$p1"; wait "$p2"
PYTHONPATH="$V20/stage_r3_probe_route" CUDA_VISIBLE_DEVICES=0 "$PY" "$V20/stage_r3_probe_route/materialize_candidate_probe_packets.py" --dataset hotpotqa --split "$RUN/protocol/hotpotqa_final_test_inputs_n300.jsonl" --profiles "$V20/stage_r3_probe_route/hotpot_transfer/resource_profiles/client_profiles.json" --local-index-root "$V17/retrieval/local_indexes/hotpotqa/topic_silo" --inherited-routes "$RUN/retrieval/hotpotqa_inherited_routes.jsonl" --output "$RUN/retrieval/hotpotqa_probe_packets.jsonl" --device cuda --resume >"$RUN/logs/hotpot_packets.log" 2>&1

# Phase 1b: frozen centralized references, also query/corpus only.
central() {
  local ds="$1" split="$2" index="$3" gpu="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$V17/retrieval/01_generate_federated_pools.py" --dataset "$ds" --split "$split" --index "$index" --output "$RUN/retrieval/${ds}_centralized.jsonl" --partition centralized --centralized --pool-size 10 --sparse-candidates 5000 --device cuda >"$RUN/logs/${ds}_central.log" 2>&1
}
central hotpotqa "$RUN/protocol/hotpotqa_final_test_inputs_n300.jsonl" "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite" 0 & c0=$!
central 2wikimultihopqa "$RUN/protocol/2wikimultihopqa_final_test_inputs_n300.jsonl" "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/2wikimultihopqa.sqlite" 1 & c1=$!
central musique "$RUN/protocol/musique_final_test_inputs_n300.jsonl" "$ROOT/V7-HP-PAPER/v16_action_composition/retrieval/indexes/musique.sqlite" 2 & c2=$!
wait "$c0"; wait "$c1"; wait "$c2"

context() {
  local ds="$1" model="$2" index="$3"
  "$PY" "$STAGE/materialize_r5_contexts.py" --dataset "$ds" --split "$RUN/protocol/${ds}_final_test_inputs_n300.jsonl" --packets "$RUN/retrieval/${ds}_probe_packets.jsonl" --model "$model" --index "$index" --central-pool "$RUN/retrieval/${ds}_centralized.jsonl" --output "$RUN/retrieval/${ds}_contexts_unlabeled.jsonl"
}
context hotpotqa "$V20/stage_r3_probe_route/hotpot_transfer/models/logistic_seed_20260807.pkl" "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite"
context 2wikimultihopqa "$V20/stage_r3_probe_route/ranker_training/models/2wikimultihopqa/logistic_seed_20260807.pkl" "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/2wikimultihopqa.sqlite"
context musique "$V20/stage_r3_probe_route/ranker_training/models/musique/logistic_seed_20260807.pkl" "$ROOT/V7-HP-PAPER/v16_action_composition/retrieval/indexes/musique.sqlite"
cat "$RUN/retrieval/hotpotqa_contexts_unlabeled.jsonl" "$RUN/retrieval/2wikimultihopqa_contexts_unlabeled.jsonl" "$RUN/retrieval/musique_contexts_unlabeled.jsonl" >"$RUN/retrieval/retrieval_outputs_unlabeled.jsonl"
sha256sum "$RUN/retrieval/retrieval_outputs_unlabeled.jsonl" >"$RUN/checksums/retrieval_outputs_unlabeled.sha256"

# Phase 2: both readers complete before the evaluation process may start.
CUDA_VISIBLE_DEVICES=0 "$PY" "$STAGE/run_r5_reader_unscored.py" --reader flan --contexts "$RUN/retrieval/retrieval_outputs_unlabeled.jsonl" --sample-root "$RUN/protocol" --v16-eval "$V16E" --output "$RUN/reader_predictions/flan_unscored.jsonl" --device cuda --batch-size 4 --resume >"$RUN/logs/flan_unscored.log" 2>&1 & r0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$STAGE/run_r5_reader_unscored.py" --reader unifiedqa --contexts "$RUN/retrieval/retrieval_outputs_unlabeled.jsonl" --sample-root "$RUN/protocol" --v16-eval "$V16E" --output "$RUN/reader_predictions/unifiedqa_unscored.jsonl" --device cuda --batch-size 4 --resume >"$RUN/logs/unifiedqa_unscored.log" 2>&1 & r1=$!
wait "$r0"; wait "$r1"

# Phase 3/4: evaluator validates all 7,200 unscored records, freezes checksums, then opens labels once.
"$PY" "$STAGE/evaluate_r5_after_unseal.py" --root "$ROOT" --run "$RUN" --v17 "$V17" --v16-eval "$V16E" --r4-main "$V20/stage_r4_frozen_reader/statistics/main_reader_results.csv" --r4-bootstrap "$V20/stage_r4_frozen_reader/statistics/paired_bootstrap.csv" >"$RUN/logs/final_evaluation.log" 2>&1

"$PY" - "$RUN" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); output={}
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'artifact_checksum_manifest.json':
        h=hashlib.sha256(path.read_bytes()).hexdigest(); output[str(path.relative_to(root))]={"sha256":h,"bytes":path.stat().st_size}
(root/'checksums/artifact_checksum_manifest.json').write_text(json.dumps({"status":"v20_empirical_evaluation_complete","files":output},indent=2)+'\n')
PY
echo "R5_COMPLETE"
