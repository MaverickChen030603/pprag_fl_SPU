# ARS Plan: Significance-Forward but Evidence-Faithful Revision

## Paper configuration

- Venue and type: SIGIR-AP full paper, nine-page main text.
- Revision mode: communication and argument revision only; no model, threshold, split, or metric changes.
- Primary reader takeaway: Full demonstrates replicated reader-level benefit from opportunity-aware selective context construction, while matched reranking reveals a meaningful answer-evidence-cost choice.
- Integrity constraint: every absolute score, delta, confidence interval, latency, and post-hoc label remains available in its proper table or boundary section.

## Chapter plan

1. **Abstract:** open with the multi-hop context problem and replicated result; describe CrossEncoder as a complementary operating point; close with the framework contribution. Avoid leading with limitations.
2. **Introduction:** move from compositional context need to candidate availability, then selective realization. Present the CrossEncoder comparison as the experiment that identifies the multi-objective shape of the problem.
3. **Related Work:** distinguish retrieval, independent reranking, set/context construction, compression, and selective prediction. Position the paper at their intersection without claiming that prior work ignores candidates.
4. **Method/Protocol:** retain the no-leak contract as a credibility asset. Define risk-controlled once and separate empirical calibration from guarantees.
5. **Results:** lead with the replicated positive direction across all three metrics; keep absolute values in tables. Use CrossEncoder to show operating points, then oracle decomposition to show improvement headroom.
6. **Mechanism/Cost:** connect pair/chain ablations to action opportunity, not universal necessity. State all-query computation clearly.
7. **Boundary/Limitations:** concentrate 2Wiki, bounded-pool, hardware, and cost limits here instead of repeating them in Abstract and every result paragraph.
8. **Conclusion:** end on the contribution: a framework and frozen empirical account of availability, realization, and answer-evidence-cost trade-offs.

## Significance hierarchy

1. Replication across two frozen holdouts with positive Answer/SP/Joint movement.
2. Fully nested and leak-controlled selective-policy evaluation.
3. First-class separation of absent opportunities and selection misses within frozen actions.
4. Protocol-matched evidence that Full and CrossEncoder occupy different evaluated operating points.
5. Explicit measured risk and cost boundaries.

## Language policy

- Prefer: `consistent`, `replicated`, `statistically reliable`, `bounded`, `operating point`, `unrealized utility`, `headroom`.
- Avoid: `tiny`, `failure list`, `universally superior`, `safe`, `guaranteed`, `efficient`, `SOTA`.
- Do not hide: absolute metrics, confidence intervals, Answer-drop, latency, or non-significant transfer.
