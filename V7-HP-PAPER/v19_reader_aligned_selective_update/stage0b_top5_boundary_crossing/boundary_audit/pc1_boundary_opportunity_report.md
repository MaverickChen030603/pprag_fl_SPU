# PC-1 Boundary Opportunity Audit

Queries audited: 100
Top-10/20 changed queries: 24
Support-promoting among Top-10/20 changed: 0 / 24
Queries with support already in rank 6-10: 12
BoundaryOpportunity@5 (threshold=0.02): 4 / 100
BoundaryConversionRate: 0 / 12
UsefulTop5ChangeRate: 0 / 100
HarmfulTop5ChangeRate: 0 / 100
Support rank improved/worsened documents: 2 / 0

## Category Counts

| Category | Count |
|---|---:|
| C_deep_support | 45 |
| D_irrelevant_reorder | 55 |

## Answers Required by Stage 0B-1

1. In the Top-10/20 changed set, 0 of 24 queries show support-promoting movement or support-score gain.
2. 12 queries have at least one gold/support document at baseline rank 6-10.
3. The rank-5 boundary appears usable only when BoundaryOpportunity@5 is non-trivial; inspect `rank5_rank10_margin.csv` before training.
4. The two Top-5 change cases are summarized by category: useful=0, harmful=0, other=2.

No reader gate decision is made by this audit.
