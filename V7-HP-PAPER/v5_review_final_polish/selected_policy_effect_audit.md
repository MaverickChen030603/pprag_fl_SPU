# Selected Policy Effect Audit

## Frozen decomposition

| Holdout | Coverage | Metric | Population delta | Selected mean | Wins/Losses/Ties | Selected drop | Median [Q25, Q75] | Fallback |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 3,000 | 774/3000 (25.8%) | Answer F1 | +0.0088 | +0.0340 | 89/60/625 | 7.75% | 0 [0, 0] | Exactly zero delta |
| 3,000 | 774/3000 (25.8%) | SP F1 | +0.0056 | +0.0219 | 123/100/551 | 12.92% | 0 [0, 0] | Exactly zero delta |
| 3,000 | 774/3000 (25.8%) | Joint F1 | +0.0064 | +0.0250 | 141/115/518 | 14.86% | 0 [0, 0] | Exactly zero delta |
| 3,405 | 881/3405 (25.9%) | Answer F1 | +0.0116 | +0.0447 | 107/69/705 | 7.83% | 0 [0, 0] | Exactly zero delta |
| 3,405 | 881/3405 (25.9%) | SP F1 | +0.0061 | +0.0237 | 127/94/660 | 10.67% | 0 [0, 0] | Exactly zero delta |
| 3,405 | 881/3405 (25.9%) | Joint F1 | +0.0080 | +0.0309 | 169/125/587 | 14.19% | 0 [0, 0] | Exactly zero delta |

## Interpretation rule

Although the selected subset has larger mean deltas, most selected contexts tie the baseline and some are harmful; the conditional result characterizes the policy's chosen subset rather than an oracle-improvable population.

The previously circulated 2.0% same-source answer-drop figure is an aggregate/population quantity and is outdated for the selected-policy risk claim. The final manuscript consistently uses the direct selected-query rates of 7.75% and 7.83%. No selected-query mean is described as a causal effect, a guarantee, or an effect on all improvable queries.
