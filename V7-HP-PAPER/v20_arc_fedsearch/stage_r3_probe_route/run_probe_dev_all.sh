#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
STAGE="$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route"

"$STAGE/run_probe_dev_dataset.sh" 2wikimultihopqa 0
"$STAGE/run_probe_dev_dataset.sh" musique 0
"$PY" "$STAGE/summarize_probe_dev.py" --stage-root "$STAGE"
