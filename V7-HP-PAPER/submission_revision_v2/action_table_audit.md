# Action Table Consistency Audit

## Scope reconciliation

The action table has five logical candidate rows per query, hence 5,000 rows for 1,000 queries. One logical single-insertion generator emits the fallback name `keep_top3_insert1_slot5` for six no-op cases, so six stored `candidate_name` values appear even though there are five generated rows per query.

| Logical template | Stored rows | Paper-positive rows | Effective rows | Eligible in main selector | Nested selected count | Placement |
|---|---:|---:|---:|---|---:|---|
| `top4_bg1_balanced` | 1,000 | 70 | 809 | Yes, if effective | 22 | Main |
| `keep_top2_insert1_slot3` | 1,000 | 98 | 937 | Yes, if effective | 12 | Main |
| `keep_top3_insert1_slot4` (includes six stored slot-5 no-ops) | 1,000 | 117 | 994 | Yes, if effective | 415 | Main |
| `keep_top3_bridge_insert1` | 1,000 | 94 | 783 | Yes, if effective | 51 | Main |
| `keep_top3_insert2_strict` | 1,000 | 69 | 994 | **No** | 0 | Appendix diagnostic |
| **Total** | **5,000** | **448** | **4,517** | 4,000 materialized rows before effectiveness gating | **500** | 500 fallbacks separately |

## Correct positive-opportunity statistics

- **All materialized actions:** 448/5,000 are paper-positive; 222 queries have at least one positive; 778 have none.
- **Main-eligible logical scope (two-document insertion excluded):** 379/4,000 are paper-positive; 203 queries have at least one positive; **797 have none**.
- Excluding the two-document template removes 69 positive rows and removes the only positive option for 19 queries.

Therefore 778 is an all-action diagnostic, whereas 797 is the correct opportunity bottleneck for the submitted main selector. Figure 3 and the main analysis must use 797. The 778 statistic may appear only when explicitly labeled “all five materialized templates.”

## Fallback accounting

Fallback is not a sixth candidate action row and is not included in the 5,000-row positive rate. It is the decision to preserve the frozen baseline after score, safety, eligibility, and coverage checks. In the fully nested primary estimate, 500 queries receive an effective action and 500 use fallback.

## Selected behavior

The nested selector chooses 415 conservative slot-4 insertions, 51 bridge insertions, 22 reorderings, and 12 slot-3 insertions. It never selects the excluded two-document action. It recovers 59 selected paper-positive actions and 26.6% of positive action opportunities under the diagnostic denominator used by the training artifact.
