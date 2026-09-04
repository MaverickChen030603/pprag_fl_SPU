# Generator Component Claim Audit

| Component | Observed result | Allowed claim | Forbidden claim |
| --- | --- | --- | --- |
| Pair complementarity | Density 14.71% to 10.27% when removed; coverage 29.2% to 27.7% | Clearest learned-component contribution | Every pair score improves every metric |
| Two-document chains | Coverage 29.2% to 25.1%; non-ceiling 47.63% to 40.92% | Clearest structural contribution | More documents always help |
| Document opportunity model | Removal raises coverage to 32.6% but lowers safety to 91.74% | Trades raw breadth for answer safety | Monotonically increases opportunity |
| Lexical-only generator | Coverage 30.7%, density 13.87% | Useful diagnostic showing non-monotonic feature effects | Semantic features dominate lexical features uniformly |
| Missing-hop, MPNet, cross-encoder, redundancy | Mixed effects | Frozen recipe components; full rows in appendix | Independently necessary innovations |

The main table contains only the full generator, pair removal, chain removal, document-model removal, and lexical-only diagnostic. Remaining rows are moved to the appendix. The final papers never state that every semantic component contributes positively.
