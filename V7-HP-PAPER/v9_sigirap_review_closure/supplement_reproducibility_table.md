# Supplement Reproducibility Table

| Module | Input dimension / feature family | Model and normalization | Labels | Frozen decision rule |
| --- | --- | --- | --- | --- |
| Missing-hop | 11 query/baseline summaries | StandardScaler + balanced logistic; nested C in 0.1/1/10 | Five missing-type classes | Five probabilities |
| Document opportunity | 14 document features | Per-query CE min-max, then StandardScaler + balanced logistic | Training action usefulness | Candidate probability |
| Pair complementarity | 13 pair features | StandardScaler + balanced logistic | Training pair complementarity | Top three proposals; ten scored pairs |
| Action generator | Document, pair, missing-hop and family scores | Deterministic bounded constructor | None at inference | 4/6/8 dynamic cap, max eight plus fallback |
| Preservation head | 16 action features | StandardScaler + balanced logistic, C=.5 | Answer F1 non-decrease | Fold threshold 0.5 or 0.6 |
| Utility head | Same 16 action features | StandardScaler + balanced logistic, C=.5 | Answer-compatible utility gain | Fold threshold 0.3 |
| Coverage gate | Two probabilities | Deterministic sort | Inner OOF objective | Fold coverage 0.15/0.25/0.30 |
| Support predictor | Nine sentence features | StandardScaler + balanced logistic, C=.5 | Official supporting sentence | Threshold 0.7; top five, minimum two |

Seed for generator, selector, support model and paired bootstrap: **20260714** unless the diagnostic script explicitly records **20260715**. Exact fold configurations and checkpoint hashes are in `seed_and_checkpoint_manifest.md`.
