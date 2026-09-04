# Candidate-Pool Sensitivity and Scope

The frozen HotpotQA distractor artifact does not provide a common 20/50/100-document per-query pool. Expanding it by mixing documents from other queries would change the retrieval problem and would not be a valid sensitivity analysis. We therefore report the observed pool availability, the measured size-10 lexical cost, and the exact pair-count bound under top-L pruning; unavailable quality cells are marked rather than imputed.

| Requested pool | Eligible frozen queries | All pairs | Pairs after top-L | Quality result |
|---:|---:|---:|---:|---|
| 10 | 2973 | 45 | 45 | [NOT AVAILABLE: only the frozen size-10 retrieval pool has reader outcomes] |
| 20 | 1 | 190 | 45 | [NOT AVAILABLE] |
| 50 | 0 | 1225 | 45 | [NOT AVAILABLE] |
| 100 | 0 | 4950 | 45 | [NOT AVAILABLE] |

**Scope:** The method targets reader-facing organization over a bounded retrieved pool; corpus-scale retrieval and streaming index maintenance are outside its scope.
