#!/usr/bin/env bash
# Frozen V20 R4 pipeline. Retrieval configuration is not modified here.
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
STAGE="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
OUT="$STAGE"
LOG="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/logs"
mkdir -p "$OUT"/{protocol,input_manifests,centralized_reference,flan,unifiedqa,statistics,mechanism,gap_recovery,errors,reports} "$LOG"

"$PY" "$STAGE/preflight_r4.py" --output-root "$OUT"

# The historical centralized pools use different query IDs. This only replays
# their already-frozen global hybrid contract on the R3 holdouts; it is not a
# new federated method and is reported solely as a reference.
central() {
  local ds split index gpu output
  ds="$1"; split="$2"; index="$3"; gpu="$4"
  output="$OUT/centralized_reference/${ds}_centralized_pool.jsonl"
  if [[ ! -s "$output" || ! -s "${output%.jsonl}.manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$V17/retrieval/01_generate_federated_pools.py" \
      --dataset "$ds" --split "$split" --index "$index" --output "$output" --partition centralized --centralized \
      --pool-size 10 --sparse-candidates 5000 --device cuda >"$LOG/r4_central_${ds}.log" 2>&1
  fi
}
central hotpotqa "$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route/hotpot_transfer/protocol/probe_holdout.jsonl" "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite" 0 & p0=$!
central 2wikimultihopqa "$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route/protocol/2wikimultihopqa/probe_holdout.jsonl" "$ROOT/V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/2wikimultihopqa.sqlite" 1 & p1=$!
central musique "$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route/protocol/musique/probe_holdout.jsonl" "$ROOT/V7-HP-PAPER/v16_action_composition/retrieval/indexes/musique.sqlite" 2 & p2=$!
wait "$p0"; wait "$p1"; wait "$p2"

for ds in hotpotqa 2wikimultihopqa musique; do
  "$PY" "$STAGE/materialize_r4_contexts.py" --dataset "$ds" --central-pool "$OUT/centralized_reference/${ds}_centralized_pool.jsonl" --output "$OUT/input_manifests/${ds}_contexts.jsonl"
done
"$PY" - "$OUT/input_manifests" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); out=root/'all_contexts.jsonl'
with out.open('w',encoding='utf-8') as dst:
    for ds in ('hotpotqa','2wikimultihopqa','musique'):
        dst.write((root/f'{ds}_contexts.jsonl').read_text(encoding='utf-8'))
PY
"$PY" "$STAGE/build_r4_sample_manifest.py" --contexts "$OUT/input_manifests/all_contexts.jsonl" --output "$OUT/input_manifests/reader_sample_manifest.json"

# P1 uses a deterministically preselected 10-query/dataset engineering slice.
for reader in flan unifiedqa; do
  "$PY" "$STAGE/run_r4_reader.py" --reader "$reader" --contexts "$OUT/input_manifests/all_contexts.jsonl" --output "$OUT/${reader}/smoke_per_query.jsonl" --device cuda --batch-size 4 --smoke-per-dataset 10 >"$LOG/r4_${reader}_smoke.log" 2>&1
done

# The smoke result is never used to change any R4 setting. The formal run
# resumes safely if a maintenance event interrupts a process.
CUDA_VISIBLE_DEVICES=0 "$PY" "$STAGE/run_r4_reader.py" --reader flan --contexts "$OUT/input_manifests/all_contexts.jsonl" --output "$OUT/flan/per_query_results.jsonl" --device cuda --batch-size 4 --resume >"$LOG/r4_flan_full.log" 2>&1 & r0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$STAGE/run_r4_reader.py" --reader unifiedqa --contexts "$OUT/input_manifests/all_contexts.jsonl" --output "$OUT/unifiedqa/per_query_results.jsonl" --device cuda --batch-size 4 --resume >"$LOG/r4_unifiedqa_full.log" 2>&1 & r1=$!
wait "$r0"; wait "$r1"

"$PY" "$STAGE/analyze_r4.py" --flan "$OUT/flan/per_query_results.jsonl" --unifiedqa "$OUT/unifiedqa/per_query_results.jsonl" --output-root "$OUT"
