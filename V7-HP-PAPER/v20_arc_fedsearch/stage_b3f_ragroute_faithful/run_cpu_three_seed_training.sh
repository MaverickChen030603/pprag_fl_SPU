#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/srv/lab/projects/jiankangchen/pprag_fl_SPU}"
STAGE="${STAGE:-$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_b3f_ragroute_faithful}"
PY="${PY:-/srv/lab/projects/jiankangchen/envs/proberoute-py310/bin/python}"
for dataset in hotpotqa 2wikimultihopqa musique; do
  nice -n 10 "$PY" "$STAGE/train_b3f_ragroute.py" --dataset "$dataset" \
    --features "$STAGE/features/$dataset/candidate_rows.npz" \
    --contract "$STAGE/protocol/b3f_training_contract.json" \
    --output-dir "$STAGE/models/$dataset" --device cpu --batch-size 512
done
