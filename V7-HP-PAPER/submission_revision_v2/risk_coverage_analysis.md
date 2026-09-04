# Risk-Coverage Analysis

## Protocol status

The primary 0.5 intervention coverage is selected inside each outer training fold and frozen on its 200-query outer test split. The post-hoc curve in `risk_coverage_curve.csv` replays the frozen fold models at target coverages from 0.1 to 1.0. It is a diagnostic curve, not a second round of model selection and not a formal calibrated risk guarantee.

## Results

| Target coverage | Realized coverage | Delta answer F1 | Delta title recall | Delta title F1 | Delta product | Selected answer-drop rate | Fallback rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.100 | -0.0025 | +0.0035 | +0.0046 | +0.0016 | 0.070 | 0.900 |
| 0.2 | 0.200 | +0.0017 | +0.0095 | +0.0127 | +0.0094 | 0.070 | 0.800 |
| 0.3 | 0.300 | -0.0029 | +0.0095 | +0.0123 | +0.0037 | 0.067 | 0.700 |
| 0.4 | 0.400 | -0.0000 | +0.0095 | +0.0114 | +0.0064 | 0.062 | 0.600 |
| 0.5 | 0.500 | +0.0028 | +0.0120 | +0.0150 | +0.0079 | 0.058 | 0.500 |
| 0.6 | 0.600 | +0.0080 | +0.0140 | +0.0170 | +0.0132 | 0.057 | 0.400 |
| 0.7 | 0.700 | +0.0045 | +0.0140 | +0.0174 | +0.0121 | 0.056 | 0.300 |
| 0.8 | 0.800 | +0.0078 | +0.0150 | +0.0190 | +0.0159 | 0.054 | 0.200 |
| 0.9 | 0.875 | +0.0103 | +0.0155 | +0.0200 | +0.0199 | 0.053 | 0.125 |
| 1.0 | 0.878 | +0.0103 | +0.0155 | +0.0200 | +0.0199 | 0.053 | 0.122 |

The 0.9 and 1.0 targets cannot reach full coverage because safety/family eligibility gates reject 122-125 queries. The apparent improvement at high coverage is encouraging, but selecting 0.9 after inspecting these held-out outcomes would be post-hoc tuning. It is not promoted to the primary result.

## Interpretation

The curve does not show a simple “more intervention, more harm” pattern. Within this fixed selector, high-score edits are not the only useful ones, and the selected answer-drop rate slightly decreases as coverage increases. This may reflect conservative family gates and query-level ranking rather than calibrated uncertainty. A future experiment should pre-specify high coverage or select it entirely through an additional inner calibration loop, then evaluate once on untouched queries.

## Claim boundary

Allowed: “A diagnostic risk-coverage sweep suggests that useful actions remain beyond the 50% primary budget.”

Not allowed: “The selector is risk calibrated,” “0.9 coverage is optimal,” or “the held-out curve validates a guaranteed answer-risk level.”
