# Main-Conference Readiness Report

## Decision

**V3 status: NOT READY.** Only 2/5 readiness gates pass.

## Main result

| Method/stage | Effective actions | Positive queries | Opportunity | Title recall delta | Title F1 delta | Answer F1 delta | Product delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 fully nested selector | 4,000 | 203 | 20.3% | +0.0120 | +0.0150 | +0.0028 | +0.0079 |
| v3 bounded action generator | 7,882 | 234 | 23.4% | not selected | not selected | not selected | not selected |

The bounded generator almost doubled the number of evaluated actions, from four to 7.882 effective actions per query on average, but positive-query opportunity rose only from 20.3% to 23.4%. This is a real +3.1-point diagnostic gain, yet it misses the pre-registered 25% continuation floor and the 30% main-conference gate.

## Family analysis

| Action family | Actions | Positive actions | Positive/action | Positive queries | Query coverage |
| --- | --- | --- | --- | --- | --- |
| anchor_preserving_tail_replacement | 2,584 | 225 | 8.7% | 153 | 15.3% |
| bridge_aware_complementary_insertion | 1,988 | 137 | 6.9% | 113 | 11.3% |
| bounded_two_document_chain | 1,503 | 229 | 15.2% | 156 | 15.6% |
| redundancy_aware_replacement | 813 | 71 | 8.7% | 71 | 7.1% |
| joint_reorder_and_insert | 994 | 81 | 8.1% | 81 | 8.1% |

The bounded two-document chain contributes the most positive actions (229) and reaches 156 queries, but family coverages overlap heavily. Its effect is therefore evidence that broader action expressivity helps some cases, not evidence that candidate opportunity has been solved.

## Gates

| Gate | Status | Observed | Requirement |
| --- | --- | --- | --- |
| A_candidate_opportunity | FAIL | 23.4% | >=30% |
| B_downstream | FAIL | not run after opportunity stop | official/product/reader-consistent gain |
| C_protocol | PASS | deterministic no-leak generator; stop rule honored | no leak and train-only decisions |
| D_breadth | FAIL | scale=skipped_by_pre_registered_opportunity_gate; external=stopped_at_300_candidate_opportunity_gate | reader, scale, or positive external smoke |
| E_reproducibility | PASS | revision/environment/commands/artifacts logged | complete |

## Recommendation

Freeze the v3 result as a negative/diagnostic extension. Keep v2 as the submission fallback. A future v4 should redesign candidate *sources* or learn context construction from train-only outcomes rather than add more transformations over the same ten-document pool.
