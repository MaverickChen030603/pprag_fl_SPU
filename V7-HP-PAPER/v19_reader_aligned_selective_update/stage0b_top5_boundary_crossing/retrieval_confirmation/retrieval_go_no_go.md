# Stage 0B N=300 Retrieval Confirmation

Status: PASS

Reader may begin: yes, changed-context diagnostic may begin next

## Frozen vs PC-2B

| Metric | Frozen | PC-2B | Delta |
|---|---:|---:|---:|
| support_recall@5 | 0.681667 | 0.686667 | 0.005000 |
| complete_support@5 | 0.436667 | 0.446667 | 0.010000 |

## Context Delta Gate

| Metric | Value |
|---|---:|
| queries | 300 |
| top5_changed_rate | 0.120000 |
| top10_changed_rate | 0.240000 |
| top20_changed_rate | 0.440000 |
| support_rank_improved_queries | 40 |
| support_rank_worsened_queries | 5 |
| useful_top5_change_count | 3 |
| harmful_top5_change_count | 0 |
| complete_support_gain_count | 3 |
| complete_support_loss_count | 0 |
| boundary_conversion_count | 3 |

## Decision

PC-2B reproduces the N=100 retrieval signal on a disjoint N=300 development subset. The retrieval gate passes. Reader remains disabled until a separate changed-context diagnostic job is explicitly launched under the frozen PC-2B configuration.
