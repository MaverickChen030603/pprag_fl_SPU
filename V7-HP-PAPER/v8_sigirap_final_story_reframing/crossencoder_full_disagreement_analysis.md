# CrossEncoder-Full Disagreement Analysis

This is a post-hoc descriptive mechanism analysis over frozen per-query reader and official-metric outputs. It does not retrain either system, tune a threshold, or establish a causal effect of anchor preservation.

## CrossEncoder versus Frozen Top-5

| Split | Metric | Wins | Losses | Ties |
| --- | --- | ---: | ---: | ---: |
| Original holdout (3,000) | answer_f1 | 312 | 359 | 2329 |
| Original holdout (3,000) | sp_f1 | 850 | 760 | 1390 |
| Original holdout (3,000) | joint_f1 | 774 | 742 | 1484 |
| Revision holdout (3,405) | answer_f1 | 371 | 409 | 2625 |
| Revision holdout (3,405) | sp_f1 | 986 | 829 | 1590 |
| Revision holdout (3,405) | joint_f1 | 912 | 806 | 1687 |

## Full versus CrossEncoder

| Split | Metric | Wins | Losses | Ties |
| --- | --- | ---: | ---: | ---: |
| Original holdout (3,000) | answer_f1 | 338 | 263 | 2399 |
| Original holdout (3,000) | sp_f1 | 748 | 817 | 1435 |
| Original holdout (3,000) | joint_f1 | 726 | 738 | 1536 |
| Revision holdout (3,405) | answer_f1 | 398 | 322 | 2685 |
| Revision holdout (3,405) | sp_f1 | 829 | 942 | 1634 |
| Revision holdout (3,405) | joint_f1 | 800 | 875 | 1730 |

## Cross-events

| Split | Event | N | Proportion | Bootstrap 95% CI |
| --- | --- | ---: | ---: | --- |
| Original holdout (3,000) | CE SP up, Answer down | 63 | 2.1% | [1.6%, 2.6%] |
| Original holdout (3,000) | CE Joint up, Answer down | 5 | 0.2% | [0.0%, 0.3%] |
| Original holdout (3,000) | Full Answer up, CE Answer down | 1 | 0.0% | [0.0%, 0.1%] |
| Original holdout (3,000) | Both Answer up | 72 | 2.4% | [1.9%, 3.0%] |
| Original holdout (3,000) | Both Answer down | 39 | 1.3% | [0.9%, 1.7%] |
| Original holdout (3,000) | Both Joint up | 102 | 3.4% | [2.8%, 4.1%] |
| Revision holdout (3,405) | CE SP up, Answer down | 74 | 2.2% | [1.7%, 2.7%] |
| Revision holdout (3,405) | CE Joint up, Answer down | 9 | 0.3% | [0.1%, 0.4%] |
| Revision holdout (3,405) | Full Answer up, CE Answer down | 0 | 0.0% | [0.0%, 0.0%] |
| Revision holdout (3,405) | Both Answer up | 88 | 2.6% | [2.1%, 3.1%] |
| Revision holdout (3,405) | Both Answer down | 38 | 1.1% | [0.8%, 1.5%] |
| Revision holdout (3,405) | Both Joint up | 127 | 3.7% | [3.1%, 4.4%] |

## Full action families when Full improves Answer and CE lowers Answer

| Split | Action family | N | Share within event |
| --- | --- | ---: | ---: |
| Original holdout (3,000) | answer_anchor_first_reorder | 1 | 100.0% |

## Anchor-label boundary

The frozen artifacts contain inference-time anchor proxies but no reliable explicit answer-anchor label. We therefore do not report anchor retention or create an outcome-derived anchor label. The paired disagreement patterns are associations between system outputs, not causal proof of any particular document mechanism.
