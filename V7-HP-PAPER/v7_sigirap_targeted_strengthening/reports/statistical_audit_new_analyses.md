# Statistical Audit of New Analyses

| Check | Status | Evidence and boundary |
| --- | --- | --- |
| Oracle uses target outcomes | PASS | Utility and answer-preserving choices read per-query reader/official outcomes and are labeled retrospective diagnostics. |
| Oracle excluded from model selection | PASS | No oracle value enters action generation, Full, selector fitting, thresholding, or reranker choice. |
| CrossEncoder variant chosen on development only | PASS | Score order is selected by development Joint F1; both holdouts are evaluation-only. |
| No holdout tuning | PASS | Full, selector thresholds, support threshold, CE checkpoint, document budget, prompt, and decoding remain frozen. |
| Pair pruning is exploratory | PASS | Quality is development-only; holdout quality is not searched and no k is promoted. |
| Subgroup multiplicity | PASS | Four 2Wiki Joint-F1 subgroup p-values use Benjamini-Hochberg FDR; no group remains significant. |
| Small subgroup claims | PASS | All official groups have N>=100; effects are still described as exploratory. |
| Query pairing | PASS | Bootstrap differences are formed within query before 5,000 paired resamples. |
| Holdouts pooled | PASS | The 3,000 and 3,405 results and intervals remain separate. |
| New p-value family called pre-specified | PASS | Oracle and reranker analyses are called post-hoc; subgroup tests are exploratory. |
| Effect size with absolute score | PASS | Main and supplement tables provide baseline, secondary baseline, Full, and oracle absolute F1. |
| CI reproducibility | PASS | Scripts fix seed 20260715 and write per-query CSVs used by every interval. |
| Primary results not overwritten | PASS | Frozen +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 remain unchanged. |

The oracle and CrossEncoder analyses answer new retrospective questions after the primary study. Their p-values do not belong to one pre-specified confirmatory family and are not used to revise Full.
