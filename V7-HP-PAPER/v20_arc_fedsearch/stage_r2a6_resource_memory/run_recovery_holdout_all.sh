#!/usr/bin/env bash
# This entry point is intentionally guarded: it may run only after Dev selects one shared REMP configuration.
set -euo pipefail

ROOT="${FEDSEARCH_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PY="${FEDSEARCH_PY:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
V20="$ROOT/V7-HP-PAPER/v20_arc_fedsearch"
V17="$ROOT/V7-HP-PAPER/v17_fedaction_rag"
STAGE="$V20/stage_r2a6_resource_memory"
SELECTED_METHOD="$($PY -c 'import json,sys; x=json.load(open(sys.argv[1])); s=x.get("selected"); print(s["method"] if x.get("status")=="recovery_dev_passed_holdout_required" and s else "")' "$STAGE/reports/next_method_decision.json")"
if [[ -z "$SELECTED_METHOD" ]]; then
  echo "Recovery-Holdout is blocked: Recovery-Dev did not select an eligible shared method." >&2
  exit 3
fi

# Three construction seeds are recorded. The first is the preregistered primary;
# the other two are seed robustness checks and never select a new method.
for SEED in 20260806 20260807 20260808; do
  for DATASET in 2wikimultihopqa musique; do
    R2="$V20/stage_r2_mars_route/$DATASET"
    BEST_P=8; [[ "$DATASET" == "musique" ]] && BEST_P=16
    PROFILE="$STAGE/holdout/profiles_seed_${SEED}/$DATASET"
    OUT="$STAGE/holdout/seed_${SEED}/$DATASET"
    mkdir -p "$PROFILE" "$OUT"
    CUDA_VISIBLE_DEVICES=0 "$PY" "$STAGE/memory_profiles/build_remp_profiles.py" \
      --dataset "$DATASET" --router-train "$R2/protocol/router_train.jsonl" \
      --local-index-root "$V17/retrieval/local_indexes/$DATASET/topic_silo" \
      --output-dir "$PROFILE" --device cuda --seed "$SEED"
    CUDA_VISIBLE_DEVICES=0 "$PY" "$STAGE/candidate_generation/evaluate_recovery.py" \
      --dataset "$DATASET" --split "$STAGE/protocol/$DATASET/recovery_holdout.jsonl" \
      --legacy-profiles "$R2/resource_profiles/client_profiles.json" \
      --units "$PROFILE/client_memory_units.jsonl" --selected-embeddings "$PROFILE/selected_unit_embeddings.npz" \
      --assignment "$V17/partitions/assignments/$DATASET/topic_silo_m20.jsonl" --output-dir "$OUT" \
      --best-p "$BEST_P" --only-methods "$SELECTED_METHOD" --device cuda
  done
done

"$PY" "$STAGE/reports/aggregate_recovery.py" \
  --phase holdout --stage-root "$STAGE" --run-root "$STAGE/holdout/seed_20260806" \
  --profile-root "$STAGE/holdout/profiles_seed_20260806" --selected-method "$SELECTED_METHOD"
