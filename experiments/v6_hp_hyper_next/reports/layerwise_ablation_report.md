# Layerwise Budget Ablation Report

## Summary

- Methods: 2

| method | MRR | F1 | payload | block Jaccard | pooler ratio | layer8 ratio | entropy | MRR delta | F1 delta | diverse? | acceptable? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v6_anchor_layerwise_on | 0.8372 | 0.72 | 0.07013433411665673 | 1.0 | 0.5 | 0.46 | 0.6931471805579453 | 0.0 | 0.0 | False | True |
| v6_layerwise_off | 0.836 | 0.718 | 0.12192627406965732 | 0.8333333333333334 | 0.35714285714285715 | 0.32857142857142857 | 1.0175192670436768 | -0.0012000000000000899 | -0.0020000000000000018 | True | True |

## Decision Rules

- Meaningful selection diversity: block Jaccard <= 0.85, pooler ratio decreases by >= 20%, or entropy increases by >= 15%.
- Metric acceptable: hard_1000 MRR/F1 drop <= 0.003.
- Promising: hard_1000 MRR or F1 improves >= 0.005.
