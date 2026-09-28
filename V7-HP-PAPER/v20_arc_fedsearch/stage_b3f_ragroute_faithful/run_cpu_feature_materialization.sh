#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/srv/lab/projects/jiankangchen/pprag_fl_SPU}"
BASE="${BASE:-$ROOT/experiments/proberoute_submission_baselines_20260915}"
STAGE="${STAGE:-$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_b3f_ragroute_faithful}"
PY="${PY:-/srv/lab/projects/jiankangchen/envs/proberoute-py310/bin/python}"
REVISION="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"

for dataset in hotpotqa 2wikimultihopqa musique; do
  case "$dataset" in
    hotpotqa)
      source="$BASE/public_training_sources/hf_hotpot/hotpot_train_v1.json"
      profile="$BASE/inputs/c1_frozen_assets/V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route/hotpot_transfer/resource_profiles/client_profiles.json" ;;
    2wikimultihopqa)
      source="$BASE/public_training_sources/2wikimultihop_train.json"
      profile="$BASE/inputs/c1_frozen_assets/V7-HP-PAPER/v20_arc_fedsearch/stage_r2_mars_route/2wikimultihopqa/resource_profiles/client_profiles.json" ;;
    musique)
      source="$BASE/public_training_sources/musique_ans_v1.0_train.jsonl"
      profile="$BASE/inputs/c1_frozen_assets/V7-HP-PAPER/v20_arc_fedsearch/stage_r2_mars_route/musique/resource_profiles/client_profiles.json" ;;
  esac
  nice -n 10 "$PY" "$STAGE/prepare_b3f_training_features.py" \
    --dataset "$dataset" --source "$source" \
    --historical-manifest "$ROOT/V7-HP-PAPER/v20_arc_fedsearch/stage_c1_fresh_confirmation/pre_registration_recovery/historical_manifests/m2_probe_train_${dataset}.json" \
    --p0-profiles "$profile" \
    --feature-centroids "$BASE/runs/ragroute_b3_r5_posthoc_20260916/centroids/$dataset/source_centroids.npy" \
    --assignment "$BASE/inputs/v17/partitions/assignments/$dataset/topic_silo_m20.jsonl" \
    --v16-eval "$BASE/inputs/v16_evaluation" --output-dir "$STAGE/features/$dataset" \
    --revision "$REVISION" --device cpu --batch-size 16
done
