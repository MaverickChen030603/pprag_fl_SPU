# 2Wiki Reasoning-Type Failure Analysis

## Taxonomy audit

Grouping uses the dataset's actual `type` field: **official 2Wiki type field; no heuristic mapping**. Counts are bridge_comparison=252, comparison=252, compositional=382, inference=114. The unmapped proportion is 0.0%.

## Type-level results

| Type | N | Baseline A/SP/J | Full delta A/SP/J | Joint 95% CI | raw p | BH-FDR p | opportunity | policy coverage |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| bridge_comparison | 252 | 0.6107/0.5037/0.3065 | +0.0010/-0.0011/+0.0004 | [-0.0109, +0.0119] | 0.9268 | 0.9872 | 48.0% | 24.6% |
| comparison | 252 | 0.6814/0.6973/0.4775 | +0.0099/-0.0048/+0.0001 | [-0.0187, +0.0183] | 0.9872 | 0.9872 | 14.3% | 25.8% |
| compositional | 382 | 0.2851/0.3158/0.1011 | +0.0179/+0.0013/+0.0087 | [+0.0007, +0.0174] | 0.0340 | 0.1360 | 34.6% | 26.4% |
| inference | 114 | 0.3192/0.2739/0.0888 | -0.0090/+0.0031/-0.0015 | [-0.0077, +0.0040] | 0.6460 | 0.9872 | 24.6% | 28.1% |

## Interpretation

No reasoning-type subgroup retains a statistically resolved Joint effect after BH-FDR correction. The available taxonomy therefore does not by itself explain the aggregate transfer uncertainty.

The distribution-shift table compares Hotpot-3,000 and 2Wiki on question/document length, pool size, entity overlap, bridge frequency, pair complementarity, policy scores, and action-family frequencies. Associations with transfer behavior are descriptive and do not establish causality.
