# Review 3: Cost and Scale Reviewer

## Summary
The study measures post-retrieval cost carefully. Full is 1.52x the baseline, and the development pair-pruning curve shows that reducing pair evaluations saves little because semantic encoding dominates.

## Strengths
- Same-machine component timing with context-match audits.
- One final reader call for all online systems.
- No low-cost variant is promoted after Lite non-inferiority failure.

## Weaknesses
- Approximately ten documents is far from corpus-scale retrieval.
- Historical offline labeling/training cost is missing.
- Full's added latency may not be justified by its small population effect, especially against CE-Top5.

## Questions
How does CE-Top5 latency compare after the direct benchmark? Which components can be cached in a realistic service without changing the protocol?

## Overall score
5/10 (borderline)

## Confidence
4/5

## Recommendation
Borderline; the paper is publishable as a bounded quality-risk-cost study, not an efficiency result.
