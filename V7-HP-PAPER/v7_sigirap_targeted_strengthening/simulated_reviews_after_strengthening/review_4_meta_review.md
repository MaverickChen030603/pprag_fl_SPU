# Review 4: Meta-Review

## Summary
The revision directly addresses the strongest likely objections. It adds a matched neural reranker, a frozen-action oracle decomposition, development-only cost sensitivity, and FDR-controlled 2Wiki structure analysis. These additions improve trust but also reveal that independent relevance explains much of the SP/Joint benefit.

## Strengths
- Post-hoc status and leakage boundaries are explicit.
- Generator-versus-selector limitations are quantified.
- 2Wiki failure remains a limitation rather than being tuned away.
- The paper can remain coherent if the main claim is narrowed.

## Weaknesses
- The methodological novelty claim is weaker after the strong baseline.
- The retrospective oracle is large but not actionable.
- Cross-domain and large-pool evidence remain absent.

## Questions
Does the final abstract clearly state the CE result and avoid implying pair superiority? Is the nine-page version focused enough to keep oracle definitions and subgroup details in the supplement?

## Overall score
6/10 (weak accept)

## Confidence
4/5

## Recommendation
Weak accept if the strengthened framing is used; weak reject if the old pair-superiority narrative remains.
