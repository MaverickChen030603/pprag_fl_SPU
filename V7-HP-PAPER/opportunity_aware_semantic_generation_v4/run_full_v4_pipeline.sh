#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"

"${PYTHON_BIN}" 00_reaudit_v3_opportunity.py
"${PYTHON_BIN}" 01_build_ceiling_aware_metrics.py
"${PYTHON_BIN}" 02_build_marginal_family_coverage.py
"${PYTHON_BIN}" 03_train_semantic_candidate_generator.py --device "${V4_DEVICE:-cuda}" --batch-size "${V4_SEMANTIC_BATCH:-64}" "${V4_REUSE_CACHE:+--reuse-cache}"
"${PYTHON_BIN}" 04_generate_outer_fold_actions.py --max-actions 8
"${PYTHON_BIN}" 05_run_reader_action_outcomes.py --device "${V4_DEVICE:-cuda}" --batch-size "${V4_READER_BATCH:-16}" --resume
"${PYTHON_BIN}" 06_run_opportunity_gate.py
"${PYTHON_BIN}" 07_train_nested_selector_v4.py
"${PYTHON_BIN}" 08_run_official_hotpot_evaluation.py
"${PYTHON_BIN}" 09_run_multi_reader_evaluation.py --device "${V4_DEVICE:-cuda}"
"${PYTHON_BIN}" 10_run_scaleup_evaluation.py
"${PYTHON_BIN}" 11_run_external_dataset_validation.py
"${PYTHON_BIN}" 12_write_main_conference_paper_v4.py
