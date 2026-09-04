# Selected Intervention Success/Failure Profile

## Scope

The analysis covers 1655 interventions selected by the already frozen policy on the two Hotpot holdouts. All predictors are available before the final reader call: retrieval scores, entity/bridge overlap, pair-complementarity proxies, bounded action structure, context length, and frozen preservation/utility probabilities. Gold answer/support features, post-reader outcomes, and oracle utility are excluded as predictors. Results are explanatory only and do not retrain the selector or alter thresholds.

## Strongest standardized associations

| Outcome | Feature | Logistic coefficient | OR per SD | Spearman rho | BH-FDR p |
| --- | --- | ---: | ---: | ---: | ---: |
| answer_gain | pair_complementarity_mean | +0.367 | 1.443 | +0.091 | 0.0008 |
| answer_gain | baseline_cross_encoder_mean | -0.365 | 0.694 | -0.134 | 0.0000 |
| answer_gain | candidate_cross_encoder_max | +0.359 | 1.432 | +0.073 | 0.0080 |
| answer_gain | no_intervention_needed | -0.292 | 0.747 | -0.101 | 0.0002 |
| answer_gain | added_doc_semantic_mean | +0.205 | 1.228 | +0.033 | 0.3140 |
| joint_gain | candidate_cross_encoder_max | +0.328 | 1.388 | +0.136 | 0.0000 |
| joint_gain | position_displacement | +0.316 | 1.371 | -0.051 | 0.0798 |
| joint_gain | baseline_cross_encoder_mean | -0.307 | 0.736 | -0.152 | 0.0000 |
| joint_gain | baseline_bm25_mean | -0.307 | 0.736 | -0.095 | 0.0005 |
| joint_gain | added_doc_opportunity_mean | +0.273 | 1.314 | +0.096 | 0.0004 |

## Interpretation boundary

These coefficients describe associations within the policy-selected subset. Selection changes the feature distribution, correlated predictors make coefficients non-causal, and multiple comparisons are controlled only for the Spearman family. The profile is not used to propose a new selector in this submission cycle.
