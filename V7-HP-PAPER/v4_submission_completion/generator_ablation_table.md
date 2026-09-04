# Table 4. Fully nested generator component ablations

| Variant | Effective actions | New actions vs V3 | Positive density | Overall coverage | Non-ceiling coverage | New V3-uncovered queries | Answer-safe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full V4 generator | 7,934 | 5,655 | 14.71% | 29.2% | 47.63% | 81 | 92.66% |
| - missing-hop estimator | 7,952 | 5,619 | 14.47% | 29.0% | 47.30% | 81 | 92.81% |
| - MPNet features | 7,948 | 5,622 | 14.41% | 29.5% | 48.12% | 83 | 92.59% |
| - cross-encoder features | 7,940 | 5,691 | 14.72% | 30.6% | 49.92% | 91 | 92.57% |
| - learned document opportunity model | 7,934 | 6,484 | 14.91% | 32.6% | 53.19% | 110 | 91.74% |
| - pair complementarity | 7,934 | 5,461 | 10.27% | 27.7% | 45.17% | 71 | 93.07% |
| - two-document chain actions | 5,547 | 3,563 | 10.40% | 25.1% | 40.92% | 54 | 93.69% |
| - anchor-preserving families | 5,909 | 4,088 | 16.57% | 27.4% | 44.68% | 73 | 92.45% |
| - redundancy actions | 7,397 | 5,298 | 14.83% | 29.2% | 47.63% | 81 | 92.85% |
| Lexical-only features | 7,952 | 5,652 | 13.87% | 30.7% | 50.25% | 89 | 92.59% |
| Semantic-only features | 7,952 | 5,929 | 14.68% | 30.6% | 49.92% | 97 | 92.48% |

Learned ablations are retrained on each outer-training split and applied to the disjoint outer test fold; structural family removals reuse the frozen fold model. No outcome from the 3,000-query holdout is used. Pair complementarity and two-document actions make the clearest positive contributions. Removing the learned document opportunity model increases raw opportunity coverage to 32.6% but lowers answer safety to 91.74%; lexical-only and semantic-only variants also show that the full generator is not a post-hoc optimum for every opportunity metric. These results support the bounded semantic action space while limiting claims that every scoring submodule is independently necessary. Selector-level V2 diagnostics are reported separately in the appendix because they use a different action table and coverage and therefore are not V4 component ablations.
