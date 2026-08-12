#!/usr/bin/env bash
# Resume the sealed R5 run after infrastructure maintenance.
# Completed Phase-1a packets are immutable; incomplete non-resumable files are archived and rebuilt.
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
STAGE="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r5_final_test"
RUN="$STAGE/run_20260812"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V16E="$ROOT/V7-HP-PAPER/v16_action_composition/evaluation"
CHECKPOINT="$RUN/maintenance_checkpoint"

if [[ -e "$RUN/reports/r5_final_decision.json" ]]; then
  echo "R5 already evaluated; refusing maintenance resume" >&2
  exit 4
fi
mkdir -p "$CHECKPOINT" "$RUN/logs" "$RUN/checksums"

# This validation reads protocol metadata and unlabeled packet files only.
"$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
audit = json.loads((run / "protocol/no_leak_audit.json").read_text())
frozen = json.loads((run / "protocol/r5_frozen_artifact_manifest.json").read_text())
if audit.get("labels_semantically_opened") is not False:
    raise RuntimeError("sealed audit no longer authorizes resume")
if frozen.get("status") != "frozen_before_retrieval":
    raise RuntimeError("frozen artifact manifest missing")
for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
    packet = run / f"retrieval/{dataset}_probe_packets.jsonl"
    count = sum(1 for line in packet.open() if line.strip())
    if count != 300:
        raise RuntimeError(f"{dataset}: immutable packet count is {count}, expected 300")
if sum(1 for line in (run / "retrieval/hotpotqa_inherited_routes.jsonl").open() if line.strip()) != 300:
    raise RuntimeError("Hotpot frozen inherited route is incomplete")
if (run / "checksums/pre_unseal_prediction_manifest.json").exists():
    raise RuntimeError("pre-unseal prediction manifest exists; one-shot resume is no longer legal")
print(json.dumps({"status": "maintenance_resume_authorized", "labels_opened": False}))
PY

archive_incomplete() {
  local path="$1" expected="$2" tag="$3" count=0
  [[ -f "$path" ]] && count=$(awk 'NF{n++} END{print n+0}' "$path")
  if [[ "$count" -eq "$expected" ]]; then
    return 0
  fi
  if [[ -f "$path" ]]; then
    local stamp
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    mv "$path" "$CHECKPOINT/${tag}.${count}rows.${stamp}.partial.jsonl"
  fi
  return 1
}

central() {
  local ds="$1" split="$2" index="$3" gpu="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$V17/retrieval/01_generate_federated_pools.py" \
    --dataset "$ds" --split "$split" --index "$index" \
    --output "$RUN/retrieval/${ds}_centralized.jsonl" --partition centralized \
    --centralized --pool-size 10 --sparse-candidates 5000 --device cuda \
    >"$RUN/logs/${ds}_central.resume.log" 2>&1
}

pids=()
if ! archive_incomplete "$RUN/retrieval/hotpotqa_centralized.jsonl" 300 hotpotqa_centralized; then
  central hotpotqa "$RUN/protocol/hotpotqa_final_test_inputs_n300.jsonl" \
    "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite" 0 & pids+=("$!")
fi
if ! archive_incomplete "$RUN/retrieval/2wikimultihopqa_centralized.jsonl" 300 2wikimultihopqa_centralized; then
  central 2wikimultihopqa "$RUN/protocol/2wikimultihopqa_final_test_inputs_n300.jsonl" \
    "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/2wikimultihopqa.sqlite" 1 & pids+=("$!")
fi
if ! archive_incomplete "$RUN/retrieval/musique_centralized.jsonl" 300 musique_centralized; then
  central musique "$RUN/protocol/musique_final_test_inputs_n300.jsonl" \
    "$ROOT/V7-HP-PAPER/v16_action_composition/retrieval/indexes/musique.sqlite" 2 & pids+=("$!")
fi
for pid in "${pids[@]}"; do wait "$pid"; done

context() {
  local ds="$1" model="$2" index="$3"
  local output="$RUN/retrieval/${ds}_contexts_unlabeled.jsonl"
  if archive_incomplete "$output" 1200 "${ds}_contexts_unlabeled"; then
    return 0
  fi
  "$PY" "$STAGE/materialize_r5_contexts.py" --dataset "$ds" \
    --split "$RUN/protocol/${ds}_final_test_inputs_n300.jsonl" \
    --packets "$RUN/retrieval/${ds}_probe_packets.jsonl" --model "$model" --index "$index" \
    --central-pool "$RUN/retrieval/${ds}_centralized.jsonl" --output "$output"
}
context hotpotqa "$V20/stage_r3_probe_route/hotpot_transfer/models/logistic_seed_20260807.pkl" \
  "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite"
context 2wikimultihopqa "$V20/stage_r3_probe_route/ranker_training/models/2wikimultihopqa/logistic_seed_20260807.pkl" \
  "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/2wikimultihopqa.sqlite"
context musique "$V20/stage_r3_probe_route/ranker_training/models/musique/logistic_seed_20260807.pkl" \
  "$ROOT/V7-HP-PAPER/v16_action_composition/retrieval/indexes/musique.sqlite"

cat "$RUN/retrieval/hotpotqa_contexts_unlabeled.jsonl" \
  "$RUN/retrieval/2wikimultihopqa_contexts_unlabeled.jsonl" \
  "$RUN/retrieval/musique_contexts_unlabeled.jsonl" >"$RUN/retrieval/retrieval_outputs_unlabeled.jsonl"
test "$(awk 'NF{n++} END{print n+0}' "$RUN/retrieval/retrieval_outputs_unlabeled.jsonl")" -eq 3600
sha256sum "$RUN/retrieval/retrieval_outputs_unlabeled.jsonl" >"$RUN/checksums/retrieval_outputs_unlabeled.sha256"

# Resume-safe unscored reader inference. The evaluator still cannot start before both waits pass.
CUDA_VISIBLE_DEVICES=0 "$PY" "$STAGE/run_r5_reader_unscored.py" --reader flan \
  --contexts "$RUN/retrieval/retrieval_outputs_unlabeled.jsonl" --sample-root "$RUN/protocol" \
  --v16-eval "$V16E" --output "$RUN/reader_predictions/flan_unscored.jsonl" \
  --device cuda --batch-size 4 --resume >"$RUN/logs/flan_unscored.log" 2>&1 & r0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$STAGE/run_r5_reader_unscored.py" --reader unifiedqa \
  --contexts "$RUN/retrieval/retrieval_outputs_unlabeled.jsonl" --sample-root "$RUN/protocol" \
  --v16-eval "$V16E" --output "$RUN/reader_predictions/unifiedqa_unscored.jsonl" \
  --device cuda --batch-size 4 --resume >"$RUN/logs/unifiedqa_unscored.log" 2>&1 & r1=$!
wait "$r0"; wait "$r1"

"$PY" "$STAGE/evaluate_r5_after_unseal.py" --root "$ROOT" --run "$RUN" --v17 "$V17" \
  --v16-eval "$V16E" --r4-main "$V20/stage_r4_frozen_reader/statistics/main_reader_results.csv" \
  --r4-bootstrap "$V20/stage_r4_frozen_reader/statistics/paired_bootstrap.csv" \
  >"$RUN/logs/final_evaluation.log" 2>&1

"$PY" - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); output = {}
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "artifact_checksum_manifest.json":
        output[str(path.relative_to(root))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
(root / "checksums/artifact_checksum_manifest.json").write_text(
    json.dumps({"status": "v20_empirical_evaluation_complete", "files": output}, indent=2) + "\n")
PY
echo "R5_COMPLETE_AFTER_MAINTENANCE"
