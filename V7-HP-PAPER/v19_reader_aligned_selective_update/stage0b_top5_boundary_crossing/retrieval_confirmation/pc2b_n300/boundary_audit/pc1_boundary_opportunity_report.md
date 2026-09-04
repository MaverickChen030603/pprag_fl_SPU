# PC-1 Boundary Opportunity Audit

Queries audited: 300
Top-10/20 changed queries: 167
Support-promoting among Top-10/20 changed: 29 / 167
Queries with support already in rank 6-10: 34
BoundaryOpportunity@5 (threshold=0.02): 13 / 300
BoundaryConversionRate: 3 / 34
UsefulTop5ChangeRate: 3 / 300
HarmfulTop5ChangeRate: 0 / 300
Support rank improved/worsened documents: 42 / 5

## Category Counts

| Category | Count |
|---|---:|
| A_support_near_boundary_positive | 9 |
| B_support_near_boundary_negative | 1 |
| C_deep_support | 136 |
| D_irrelevant_reorder | 150 |
| E_top5_beneficial_swap | 3 |
| G_tie_boundary_instability | 1 |

## Answers Required by Stage 0B-1

1. In the Top-10/20 changed set, 29 of 167 queries show support-promoting movement or support-score gain.
2. 34 queries have at least one gold/support document at baseline rank 6-10.
3. The rank-5 boundary appears usable only when BoundaryOpportunity@5 is non-trivial; inspect `rank5_rank10_margin.csv` before training.
4. The two Top-5 change cases are summarized by category: useful=3, harmful=0, other=33.

No reader gate decision is made by this audit.
