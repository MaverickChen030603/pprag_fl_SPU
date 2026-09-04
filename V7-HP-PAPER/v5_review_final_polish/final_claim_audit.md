# Final Claim Audit

## Canonical claim checks

| Claim family | Frozen evidence | Final status |
|---|---|---|
| Same-source population effect | Two disjoint Hotpot holdouts; +0.0064/+0.0080 Joint F1 | Allowed as modest same-source evidence |
| Selected-policy effect | Means plus wins/losses/ties, zero median/IQR, and 7.75-7.83% Answer drop | Allowed only as conditional descriptive evidence |
| RECOMP | Budget-controlled, non-significant Joint difference versus baseline | No general ranking |
| Cost | Full 213.48 versus 140.88 ms/query, one reader call | Measured bounded latency claim |
| Transfer | 2Wiki non-significant; calibration target missed | Failed diagnostic only |
| Multi-reader | Two answer readers, one shared support predictor | Directional answer evidence only |
| Candidate scale | Roughly ten documents; ten pairs scored/query | Bounded post-retrieval scope only |

## Dangerous phrase occurrences

Every exact occurrence in the canonical main paper and appendix is listed below. Negated limitation language is retained as a boundary rather than counted as an overclaim.

| File | Section | Phrase | Sentence | Evidence | Risk | Replacement |
|---|---|---|---|---|---|---|
| paper_anonymous_review_polished.md | 9. Analysis | `independent SP replication` | It is not an independent SP replication, and the Joint direction is not independent of the shared support component. | One support predictor is shared across FLAN and UnifiedQA. | low: explicit boundary | Keep as a negated boundary. |
| paper_anonymous_review_polished.md | 10. Limitations and Ethical Considerations | `independent SP replication` | It provides directional answer-reader evidence, not independent SP replication; its Joint result also contains the shared support component. | One support predictor is shared across FLAN and UnifiedQA. | low: explicit boundary | Keep as a negated boundary. |

**Semantic overclaim violations:** 0.

The audit treats explicit negations and limitation statements as safe boundary language. It does not infer support merely from the absence of keywords; the canonical claim table above checks the main numerical and scope claims directly.
