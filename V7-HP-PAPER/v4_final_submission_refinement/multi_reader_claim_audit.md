# Multi-Reader Claim Audit

| Claim | Decision | Reason |
| --- | --- | --- |
| The same selected contexts improve answer and joint metrics for FLAN and UnifiedQA | Allowed | Both readers have positive answer/joint deltas on the frozen holdout |
| UnifiedQA provides answer-reader directional replication | Allowed | Reader changes while contexts are fixed |
| Multi-reader support robustness | Forbidden | The support predictor is shared |
| Reader-independent full-pipeline replication | Forbidden | Support prediction is not independently retrained or rerun |

Every main-text multi-reader statement is followed by the support-predictor boundary, and the UnifiedQA table row contains a footnote.
