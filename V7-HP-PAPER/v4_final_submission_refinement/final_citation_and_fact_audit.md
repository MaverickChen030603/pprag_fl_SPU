# Final Citation and Fact Audit

| Item | Verification | Result |
| --- | --- | --- |
| All manuscript citation keys resolve | Compared against `references.bib` | Pass |
| HotpotQA, 2Wiki, MuSiQue venues | ACL Anthology/TACL records | Pass |
| RECOMP status and protocol | ICLR paper, official code, author checkpoint | Pass |
| SetR status | ACL 2025 paper; claim restricted to set selection | Pass |
| Reader-Centered Passage Selection status | COLING 2025 paper; claim restricted to reader alignment | Pass |
| RankRAG status | NeurIPS 2024 paper | Pass |
| No unresolved preprint status | No unmarked preprint is used | Pass |
| No raw placeholders | Machine scan for TODO/TBD/XX/citation-needed | Pass |
| Statistical values | Regenerated from frozen JSON summaries | Pass |
| Opportunity values | Regenerated from frozen action summary | Pass |
| RECOMP token budget | Measured with frozen FLAN tokenizer | Pass |

The RECOMP claim is limited to the official-code Top-1 setting with a standardized reader adaptation. The SetR and reader-centered citations describe their published positioning without asserting unavailable implementation details.
