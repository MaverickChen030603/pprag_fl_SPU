# Pooler Dominance Ablation Report

## Summary

- Methods: 4

| method | MRR | F1 | payload | block Jaccard | pooler ratio | layer8 ratio | entropy | MRR delta | F1 delta | diverse? | acceptable? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v6_no_pooler_cap | 0.8372 | 0.72 | 0.07013433411665673 | 1.0 | 0.5 | 0.46 | 0.6931471805579453 | 0.0 | 0.0 | False | True |
| v6_pooler_cap_25 | 0.8365 | 0.72 | 0.2824664164708358 | 0.6666666666666666 | 0.0 | 0.46 | 0.6931471805579453 | -0.0007000000000000339 | 0.0 | True | True |
| v6_pooler_cap_10 | 0.8365 | 0.72 | 0.2824664164708358 | 0.6666666666666666 | 0.0 | 0.46 | 0.6931471805579453 | -0.0007000000000000339 | 0.0 | True | True |
| v6_pooler_exclude | 0.8365 | 0.72 | 0.2824664164708358 | 0.6666666666666666 | 0.0 | 0.46 | 0.6931471805579453 | -0.0007000000000000339 | 0.0 | True | True |

## Decision Rules

- Meaningful selection diversity: block Jaccard <= 0.85, pooler ratio decreases by >= 20%, or entropy increases by >= 15%.
- Metric acceptable: hard_1000 MRR/F1 drop <= 0.003.
- Promising: hard_1000 MRR or F1 improves >= 0.005.
