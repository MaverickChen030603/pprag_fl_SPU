# Review 1: Positive IR Reviewer

## Summary
The paper isolates a useful post-retrieval problem: whether a bounded pool exposes a reader-compatible multi-hop context and whether a selective policy can identify it. The two frozen same-source holdouts, exact fallback, outcome-aware decomposition, and matched CrossEncoder baseline form an unusually transparent evidence package.

## Strengths
- Fully nested training and two disjoint frozen holdouts.
- Oracle decomposition distinguishes unavailable actions from selector misses.
- Strong secondary baseline is reported even though it narrows the method claim.
- Quality, intervention harm, and latency are reported together.

## Weaknesses
- Population gains remain small.
- CrossEncoder-Top5 matches or exceeds Full on Joint, so pair construction is not a clear winner.
- The candidate pool and reader configuration are narrow.

## Questions
Can the authors explain when the answer-preserving gate is preferable to always-on CE reranking, and report whether latency includes identical reader synchronization?

## Overall score
7/10 (accept)

## Confidence
4/5

## Recommendation
Accept as a careful bounded-pool IR analysis with honest negative evidence.
