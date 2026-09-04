# V4 Submission Storyboard

## Recommended title

**Generating Reader-Compatible Context Actions for Multi-Hop Question Answering**

Alternatives:

1. Beyond Selection: Semantic Context Action Generation for Multi-Hop Question Answering
2. Opportunity-Aware Context Generation and Reader-Safe Selection for Multi-Hop QA

## One-sentence thesis

A selector cannot help when its candidate table contains no reader-compatible intervention; query-conditioned semantic action generation expands that opportunity, and fully nested risk-controlled selection converts part of it into reproducible downstream gains.

## Argument sequence

1. Multi-hop QA context failures are not only ranking errors. The available Top-5 may contain a useful answer anchor but miss a bridge, include redundant evidence, or present facts in an unhelpful order.
2. V2 shows that a reader-safe selector can choose among bounded actions, but it is limited by a 20.3% positive-query opportunity ceiling.
3. V3 nearly doubles the action count from 4,000 to 7,882 yet raises coverage by only 3.1 points and leaves positive density near 9.4%. This isolates the candidate-opportunity gap.
4. V4 predicts missing-hop structure, document opportunity, and pair complementarity, then builds at most eight bounded context actions while protecting answer anchors.
5. A separate reader-safe selector predicts answer safety and positive utility, chooses coverage using only outer-training queries, and falls back when confidence is insufficient.
6. V4 raises opportunity coverage to 29.2% and density to 14.71%, but passes only three of five pre-specified criteria.
7. On the 1,000-query fully nested development evaluation, official answer and support F1 improve significantly; joint F1 is a positive non-significant trend.
8. With every component frozen, a disjoint 3,000-query same-source holdout shows significant answer, support, and joint gains, with consistent answer/joint directions for FLAN and UnifiedQA.
9. External 2Wiki transfer is positive for answer/joint but non-significant and support-flat. RECOMP under the standardized reader is substantially below V4. These close remaining execution gaps without justifying broad generalization or SOTA claims.
10. The paper closes on the remaining boundary: improving generation efficiency and external calibration without weakening answer safety.

## Figure and table order

- Figure 1: candidate-opportunity gap and two-stage V4 pipeline.
- Table 1: V2/V3/V4 opportunity.
- Table 2: official development metrics.
- Table 3: frozen same-source holdout across two readers.
- Table 4: generator component ablations.
- Table 5: RECOMP reproduction.
- Table 6: frozen 2Wiki validation.
