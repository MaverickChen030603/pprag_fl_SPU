#!/usr/bin/env bash
set -euo pipefail

ROOT="${FEDE4RAG_ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
OUT="$ROOT/V7-HP-PAPER/submission_revision_v2"

cd "$ROOT"

# Rebuild fully nested answer-safety features and audit fold isolation.
"$PYTHON" "$OUT/nested_safe_answer_feature_generation.py"

# Fit outer-fold selectors, evaluate 1,000 held-out decisions, and produce
# significance, ablation, risk-coverage, utility, and action-scope artifacts.
"$PYTHON" "$OUT/nested_selector_training.py"

# Reconstruct source provenance, recover gold sentence labels for future
# official evaluation, and checksum the submission artifacts.
FEDE4RAG_ROOT="$ROOT" "$PYTHON" "$OUT/build_submission_manifests.py"

# The completed reader run consumed google/flan-t5-large and the fixed action
# contexts. Its exact Hub revision was not logged, so this package does not
# claim bitwise reproduction of the historical reader generation stage.
