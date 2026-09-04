# V3 Opportunity Reanalysis

## Frozen Result

- Overall positive-query coverage: **234/1000 = 23.4%**.
- V2 main-eligible coverage: **203/1000 = 20.3%**.
- Baseline ceiling queries: **389**; non-ceiling queries: **611**.
- Conditional v3 opportunity among non-ceiling queries: **234/611 = 38.3%**.
- This conditional value is retrospective and does **not** reverse the frozen v3 stop decision.

## Marginal Family Coverage

| Family | Positive actions | Unique queries | Already v2 | New vs v2 | Leave-one-out loss |
| --- | --- | --- | --- | --- | --- |
| anchor_preserving_tail_replacement | 225 | 153 | 109 | 44 | 24 |
| bounded_two_document_chain | 229 | 156 | 109 | 47 | 36 |
| bridge_aware_complementary_insertion | 137 | 113 | 76 | 37 | 6 |
| joint_reorder_and_insert | 81 | 81 | 54 | 27 | 4 |
| redundancy_aware_replacement | 71 | 71 | 45 | 26 | 8 |

Raw positive-action count is not a sufficient family contribution measure. The table distinguishes overlap from genuinely new query opportunity; the pairwise matrix is in `outputs/tables/v3_family_overlap_matrix.md`.

## Interpretation

V3 nearly doubled the action table. Its **net** coverage gain was only **31 queries** (+3.1 points). Set-level comparison finds **81 newly covered** v2-negative queries and **50 v2-covered queries not recovered by v3**, so raw coverage delta and marginal new-query coverage must be reported separately. Positive actions remain concentrated on overlapping, already-improvable queries. The next bottleneck is semantic candidate construction and query-specific opportunity creation, not selector tuning or another fixed-template expansion.
