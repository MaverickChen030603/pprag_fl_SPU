#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/lab/projects/jiankangchen/pprag_fl_SPU
BASE=$ROOT/experiments/proberoute_submission_baselines_20260915
STAGE=$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_b3f_ragroute_faithful
PY=/srv/lab/projects/jiankangchen/envs/proberoute-py310/bin/python
DEVICE="${DEVICE:-cuda}"
CODE=$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route/materialize_candidate_probe_packets.py
REV=a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
for dataset in hotpotqa 2wikimultihopqa musique; do
  if [[ "$dataset" == hotpotqa ]]; then
    profile=$BASE/inputs/c1_frozen_assets/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route/hotpot_transfer/resource_profiles/client_profiles.json
  else
    profile=$BASE/inputs/c1_frozen_assets/V7-HP-PAPER/v20_arc_fedsearch/stage_r2_mars_route/$dataset/resource_profiles/client_profiles.json
  fi
  nice -n 10 "$PY" "$CODE" --dataset "$dataset" --split "$STAGE/inputs/${dataset}_blind_n500.jsonl" --profiles "$profile" --local-index-root "$BASE/inputs/c1_frozen_assets/V7-HP-PAPER/v17_fedaction_rag/retrieval/local_indexes/$dataset/topic_silo" --output "$STAGE/retrieval/${dataset}_probe_packets.jsonl" --device "$DEVICE" --model-revision "$REV" --resume
done
