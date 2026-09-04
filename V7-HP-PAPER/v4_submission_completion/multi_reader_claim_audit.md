# Multi-Reader Claim Audit

| Claim | Status | Evidence |
| --- | --- | --- |
| Answer direction is consistent across FLAN and UnifiedQA | Allowed | Development deltas +0.0133 and +0.0129; holdout deltas +0.0088 and +0.0110 |
| Joint direction is consistent across readers | Allowed | Holdout joint F1 deltas +0.0064 and +0.0085 |
| The selected contexts transfer to a second answer reader | Allowed | Same frozen contexts evaluated by UnifiedQA-T5-Large |
| The full support pipeline independently replicates across readers | Forbidden | Both readers reuse one support predictor and threshold |
| Reader-independent support replication | Forbidden | No second independently trained support model exists |

The support F1 rows for UnifiedQA are included to compute joint metrics on the same frozen context decisions, but they are not independent support evidence. Main-text wording is restricted to: "FLAN-T5-Large and UnifiedQA-T5-Large show consistent answer and joint directions."
