# Compact Oracle Analysis

| Split | No positive action | Positive action missed | Positive action selected |
| --- | ---: | ---: | ---: |
| Original holdout (3,000) | 2,316 | 465 | 219 |
| Revision holdout (3,405) | 2,638 | 515 | 252 |

The retrospective oracle remains substantially above the frozen policy, indicating selector regret within the generated action set. It inspects target-query reader outcomes and is therefore a post-hoc diagnostic, not a deployable system or confirmatory baseline. The decomposition also shows a separate candidate-availability limitation: many queries have no positive action under the frozen training definition.

## Ratio-reference decision

Preservation-head-only, utility-head-only, and fixed-score decision variants are not added. The frozen artifacts do not store a complete pre-threshold alternative-decision protocol at the same coverage for all three variants; constructing one would require new decision rules or threshold choices after holdout access. The requested reference analysis is therefore skipped rather than tuned post hoc.
