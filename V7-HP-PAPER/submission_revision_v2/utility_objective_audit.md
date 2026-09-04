# Utility Objective Audit

## Inherited objective

The legacy action target was:

`u = delta_product + 0.8 * delta_answer + 0.3 * delta_support_recall + 0.2 * delta_title_f1`.

The repository does not preserve an independent pre-registration or derivation for the coefficients. They were not selected by the new nested protocol, and the expression partly double-counts answer and evidence because the product already contains both. It is therefore unsuitable as the primary submission objective.

## Submission-v2 primary objective

The primary selector uses a constrained two-stage rule:

1. reject actions whose predicted answer-safety probability is below the outer-train-selected threshold;
2. rank surviving actions by predicted probability of an answer-safe evidence/product gain;
3. apply only the outer-train-selected query coverage;
4. fall back to the baseline if no action passes.

This rule encodes the paper's priority without hidden scalar trade-offs. It also makes fallback an explicit safety operation.

## Nested ablation result

| Variant | Delta answer F1 | Delta title recall | Delta title F1 | Delta product | Selected answer-drop rate |
|---|---:|---:|---:|---:|---:|
| Primary constrained two-stage | +0.0028 | +0.0120 | +0.0150 | +0.0079 | 0.058 |
| Without nested safety feature | -0.0029 | +0.0130 | +0.0160 | +0.0062 | 0.062 |
| Without support features | -0.0023 | +0.0075 | +0.0094 | +0.0056 | 0.064 |
| Inherited weighted utility | -0.0031 | +0.0150 | +0.0193 | +0.0089 | 0.062 |

The weighted diagnostic attains larger evidence and product deltas but changes mean answer F1 negatively. The constrained primary sacrifices some evidence gain to preserve the sign of answer F1. This is the intended reader-safe trade-off.

## Weight sensitivity

The train-only sensitivity grid varies answer weight over 0.5/0.8/1.0, support-recall weight over 0.1/0.3/0.5, and title-F1 weight over 0.1/0.2/0.3. It is exported as `utility_weight_sensitivity.json`. No held-out result from this grid is used to replace the primary configuration.

Across the 27 diagnostics, all selected scorers remain listwise weighted rankers, although fold-level family restrictions can change. Held-out answer-F1 deltas range from -0.0073 to +0.0001, product deltas from +0.0042 to +0.0117, title-recall deltas from +0.0130 to +0.0165, and title-F1 deltas from +0.0169 to +0.0219. This pattern reinforces the audit decision: weighted evidence utilities are robustly evidence-seeking but do not reliably protect answer quality.

## Claim boundary

The paper may say that constrained safety gating is empirically preferable for answer preservation. It may not claim that the legacy coefficients are optimal, calibrated, or theoretically derived.
