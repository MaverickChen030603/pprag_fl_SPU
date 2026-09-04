# V15 Baseline Contract Matrix

Numerical comparison is allowed only when the method consumes the same frozen
pool, emits the same five-document budget, uses the same final reader, performs
no holdout tuning, and adds no unmatched online LLM call.

| Method | Same frozen pool | Five-doc output | Same reader | Extra online reader/LLM | V15 role | Status |
|---|---:|---:|---:|---:|---|---|
| Frozen Top-5 | Yes | Yes | Yes | No | primary baseline | implementation available |
| Source-order truncation | Yes | Yes | Yes | No | simple baseline | implementation available |
| CrossEncoder-Top5 | Yes | Yes | Yes | No | strong relevance baseline | smoke reproduced |
| V14 Full | Requires adapter to new pool | Yes | Yes | No | inherited method | pending matched reproduction |
| RECOMP | Yes | representation may vary | Yes | No | compression baseline | pending matched reproduction |
| Marginal utility selector | Yes | Yes | Yes | Offline labels only | direct baseline | pending |
| Direct-Joint-delta MLP | Yes | Yes | Yes | Offline labels only | scorer baseline | implemented, labels pending |
| Fixed-pool subset utility | Yes | Yes | Yes | Offline labels only | search baseline | pending |
| V15 complete-sequence repair | Yes | Yes | Yes | Offline labels only | proposed method | search smoke passed |
| SetR | Contract audit pending | Pending | Pending | Pending | related/possible baseline | no numerical claim yet |
| Contextual Passage Utility | Contract mismatch likely | Yes | reader/API-dependent | likely | related work | no style reimplementation |
| Influence Guided Context Selection | Contract audit pending | Pending | Pending | Pending | related work | official artifact audit pending |
| Context-Picker | Contract mismatch | variable/minimal subset | model-dependent | training/LOO pipeline | related work | no style reimplementation |
| R-CPS | Different reported pool/reader loop | Yes | No | Yes | related work | contract-only |
| RankRAG | Changes model/training contract | variable | No | model training | related work | contract-only |

The rows marked pending must be resolved against official papers and code before
the baseline list is frozen for the main experiment. A method name in this table
does not imply successful reproduction or fair numerical comparability.

