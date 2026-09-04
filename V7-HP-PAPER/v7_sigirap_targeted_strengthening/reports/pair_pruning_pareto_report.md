# Development-Only Pair-Pruning Pareto Analysis

This optional analysis changes only the number of retained pair evaluations. Document features, frozen selector models, selector thresholds/coverage, reader outcomes, and action families are unchanged. Quality is evaluated only on the fully nested 1,000-query development outputs. Latency is an auditable component-scaled estimate from the frozen same-machine benchmark: pair-feature construction and pair scoring are scaled by k/10; all other measured components are held fixed.

| k | Opportunity | Positive density | Policy coverage | Answer F1 | SP F1 | Joint F1 | Joint delta | Answer-drop | Generator ms | Total ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.0% | 13.2% | 26.0% | 0.6227 | 0.4976 | 0.3297 | +0.0056 | 1.4% | 68.19 | 211.63 |
| 2 | 28.6% | 14.0% | 26.0% | 0.6227 | 0.4967 | 0.3289 | +0.0048 | 1.3% | 68.39 | 211.83 |
| 3 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 68.60 | 212.04 |
| 5 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 69.01 | 212.45 |
| 7 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 69.43 | 212.86 |
| 10 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 70.05 | 213.48 |

The frozen constructor emits at most three ranked pair-chain action slots, so k>3 can reduce neither action diversity nor quality in this replay; those rows only expose the measured pair-scoring cost slope. This is an exploratory sensitivity analysis. No k is promoted as a new primary method because both existing holdouts have already been observed and no independent non-inferiority test remains.
