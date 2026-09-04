# Score Mode Ablation Report

## Summary

- Methods: 4

| method | MRR | F1 | payload | block Jaccard | pooler ratio | layer8 ratio | entropy | MRR delta | F1 delta | diverse? | acceptable? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v6_score_value | 0.8372 | 0.72 | 0.07013433411665673 |  |  |  |  | 0.0 | 0.0 | False | True |
| v6_score_downstream_value | 0.8372 | 0.72 | 0.07013433411665673 |  |  |  |  | 0.0 | 0.0 | False | True |
| v6_score_delta | 0.8373 | 0.72 | 0.07013433411665673 |  |  |  |  | 9.999999999998899e-05 | 0.0 | False | True |
| v6_score_grad_norm | 0.8373 | 0.72 | 0.07013433411665673 |  |  |  |  | 9.999999999998899e-05 | 0.0 | False | True |

## Decision Rules

- Meaningful selection diversity: block Jaccard <= 0.85, pooler ratio decreases by >= 20%, or entropy increases by >= 15%.
- Metric acceptable: hard_1000 MRR/F1 drop <= 0.003.
- Promising: hard_1000 MRR or F1 improves >= 0.005.
