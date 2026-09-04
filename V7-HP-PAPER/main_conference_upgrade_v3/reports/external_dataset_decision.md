# External Dataset Decision

Dataset: **2WikiMultiHopQA dev-300**

- Positive actions beyond the strong BM25 baseline: 73/300 (24.33%)
- Required to continue: 25.00%
- No-leak candidate path available: True
- Adapter and sentence-support metrics available: True

Decision: **stopped_at_300_candidate_opportunity_gate**. The opportunity rate misses the gate by 0.67%. Existing reader-backed evidence remains a lexical-routing sanity check and a cross-dataset selector limitation, not a generalization result.
