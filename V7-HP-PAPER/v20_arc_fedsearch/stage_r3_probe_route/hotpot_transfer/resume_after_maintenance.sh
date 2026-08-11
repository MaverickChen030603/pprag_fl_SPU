#!/usr/bin/env bash
# Idempotent recovery for the frozen R3-T/R3-C retrieval-only pipeline.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <cuda-device>" >&2
  exit 2
fi

DEVICE="$1"
ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
STAGE="$V20/stage_r3_probe_route"
TRANSFER="$STAGE/hotpot_transfer"
DECISION="$TRANSFER/holdout/transfer_decision/reader_gate_decision.json"
LATENCY="$TRANSFER/holdout/cost_matched/latency_main_results.csv"

cd "$ROOT"
echo "recovery_commit=$(git rev-parse HEAD)"
echo "recovery_expected_minimum=05d3e0d"

# The base runner is resumable: packet manifests are written only after a
# completed materialization, while partial JSONL output is resumed safely.
if [[ ! -s "$DECISION" ]]; then
  bash "$TRANSFER/run_hotpot_transfer.sh" "$DEVICE"
fi

# Keep the latency audit independent of labels and readers. It is the only
# post-retrieval step required before the transfer decision is regenerated.
if [[ ! -s "$LATENCY" ]]; then
  PYTHONPATH="$STAGE" CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" "$TRANSFER/benchmark_cost_matched_latency.py" \
    --split "$TRANSFER/protocol/probe_holdout.jsonl" \
    --profiles "$TRANSFER/resource_profiles/client_profiles.json" \
    --local-index-root "$ROOT/V7-HP-PAPER/v17_fedaction_rag/retrieval/local_indexes/hotpotqa/topic_silo" \
    --probe-packets "$TRANSFER/packets/probe_holdout.jsonl" \
    --output-dir "$TRANSFER/holdout/cost_matched" --device cuda
fi

PYTHONPATH="$STAGE" "$PY" "$TRANSFER/summarize_hotpot_transfer.py" \
  --transfer-root "$TRANSFER" --output-dir "$TRANSFER/holdout/transfer_decision"

echo "recovery_complete=true"
echo "reader_started=false"
echo "final_test_accessed=false"
echo "Reader remains an explicit next-stage action; this script never launches it."
