# Experiment Section Draft

## 1. Dataset and Metrics

The main experiment uses HotpotQA because it provides answer, support, and joint metrics needed to evaluate no-leak action selection. We report answer_f1, joint_f1, support_recall@5, and sp_f1.

## 2. Baselines and Variants

We compare the final answer-neutral positive-action selector against v2.2 support-first selection, two-stage selection, paper-positive classification, safety removal, support-feature removal, and diagnostic oracle upper bounds.

## 3. Main HotpotQA Result

Under strict no-leak query-level cross-fitting, v2.3 improves joint_f1 by +0.0150 and improves support-side metrics, while answer_f1 remains slightly positive but not statistically significant. Support_recall@5 improves by +0.0190 and sp_f1 improves by +0.0254.

## 4. Ablation Study

The ablation study shows that support-first and safety-free variants are insufficient for the final claim. The safety predictor helps preserve answer quality, while support/routing features help convert action utility into joint/support gains.

## 5. Stability and Sensitivity

Fold-level and calibration summaries are included in the appendix. They show that the result is not solely a single-threshold artifact, while also revealing fold-level variability that should be acknowledged.

## 6. 2Wiki Diagnostic

We report 2Wiki as a diagnostic external check rather than a main generalization result. The adapter and reader pipeline transfer, but selector-level transfer beyond a strong BM25 baseline remains limited by candidate exposure, feature detectability, and safety calibration.

## 7. Failure Analysis

Failure analysis shows that the remaining ceiling is driven by candidate-pool limitations, missed positive actions, and cases where support gains do not fully translate into joint reader gains.
