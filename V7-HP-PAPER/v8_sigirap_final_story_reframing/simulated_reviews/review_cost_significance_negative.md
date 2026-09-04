# Review: Cost and Significance Negative

## Summary
Full improves the frozen baseline but adds 72.60 ms/query and modifies only about 26% of contexts while executing its generator for every query. Pair pruning barely changes total latency, and Lite fails non-inferiority.

## Correctness
High. Cost is measured consistently and its exclusions are stated.

## Novelty
Moderate, but not enough to offset the weak efficiency profile.

## Significance
Low. The practical effect is small relative to cost and is limited to a roughly ten-document pool.

## Clarity
High. The paper correctly distinguishes selective modification from selective computation.

## Reproducibility
Good. Historical offline GPU-hours are missing, but online timing is auditable.

## Topic fit
Appropriate for applied IR, though scale evidence is limited.

## Baseline fairness
Strong for CrossEncoder; RECOMP remains only an approximate budget control.

## Practical value
Low under the measured configuration. No production, energy, or alternative-hardware claim is supported.

## Key questions
- Can semantic features be cached without changing the stated protocol?
- Is there an untouched split for a genuinely lower-cost selector?

## Overall score
4/10 (reject)

## Confidence
4/5

## Recommendation
Reject for limited significance and cost, not for methodological invalidity.
