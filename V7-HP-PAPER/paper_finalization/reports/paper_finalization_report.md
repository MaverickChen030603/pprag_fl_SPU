# V7-HP-PAPER Paper Finalization Report

## 1. Executive Summary

selector_v2.3 should be frozen as the paper main result. No further selector tuning is recommended before drafting, unless reviewers or coauthors require an additional robustness check.

## 2. Final Main Result

- answer_f1_delta: 0.0023
- joint_f1_delta: 0.0150
- support_recall_delta: 0.0190
- sp_f1_delta: 0.0254
- positive_candidate_recall: 0.3288
- gate_pass: True
- paper_main_recommended: True

## 3. Why v2.3 Is Paper-Ready

v2.3 is the first no-leak cross-fitted selector in this sequence that simultaneously preserves answer_f1, improves joint_f1, improves support-side metrics, keeps selected actions effective, and improves positive candidate recall beyond v2.2.

## 4. Statistical Significance

joint_f1 is significant (p=0.0245); support_recall@5 and sp_f1 are significant. answer_f1 is positive but not significant, so we use answer-preserving language.

## 5. Ablation Evidence

The final mixed two-stage/pairwise configuration is strongest on joint_f1 and paper-main criteria. Simpler classifiers pass gate but are weaker; answer-drop rejector alone is insufficient.

## 6. No-Leak / Cross-Fit Audit

The audit is stored at `outputs/audit/no_leak_crossfit_audit.md`. It verifies disjoint query folds, train-only calibration, formal/oracle separation, and no gold answer/support inference features by artifact/source review.

## 7. Candidate Pool Limitation

778 / 1000 queries have no paper-positive action. This is the main ceiling on further selector improvements.

## 8. Feature Importance

Feature importance diagnostics are stored at `outputs/diagnostics/positive_feature_importance.json` and summarized in `outputs/tables/positive_feature_importance_table.md`.

## 9. Case Studies

Case studies are exported under `outputs/case_studies/`, separated into success, answer-neutral, and failure cases.

## 10. Failure Analysis

Failures are dominated by missing positive candidates and positive actions not selected. This supports a limitation-aware paper narrative.

## 11. Paper Claim Boundary

The claim boundary memo is stored at `reports/paper_claim_boundary_memo.md`. The central claim should be significant joint/support gains under strict no-leak cross-fitting while preserving answer_f1.

## 12. Recommended Next Writing Steps

Use the generated main result table, ablation table, no-leak audit, candidate pool breakdown, feature importance table, and case studies to draft the experiment section. Avoid launching v2.4 tuning before writing.
