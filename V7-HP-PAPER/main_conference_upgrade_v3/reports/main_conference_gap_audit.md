# Main-Conference Gap Audit

## Frozen v2 result

The existing `submission_revision_v2/` is frozen as the Findings/COLING fallback. Its files were read only and fingerprinted; v3 is a separate experiment directory.

| Requirement | Current v2 | Main-conference need | Planned action |
| --- | --- | --- | --- |
| fully nested evaluation | complete | complete | freeze |
| significant title metrics | complete | insufficient alone | preserve |
| significant answer/product gain | absent | strongly preferred | candidate redesign |
| official support/joint metrics | absent | strongly preferred | official pipeline |
| positive-query coverage | 20.3% | too low | new action generator |
| multi-reader | absent | desirable | frozen-context replay |
| scale | 1000 | limited | full/large validation |
| second dataset | failed diagnostic | desirable | gated retry |
| exact strong baselines | partial proxies | desirable | controlled faithful baselines |
| reproducibility | mostly complete | complete | archive manifest |

## Primary diagnosis

**The primary main-conference bottleneck is candidate opportunity, not selector capacity.** In the frozen main-eligible action table, only 203/1,000 queries expose at least one answer-safe positive action; 797 expose none. A stronger selector cannot select an action that does not exist.

## Frozen claim boundary

Fully nested reader-safe action selection improves title-level evidence coverage without a demonstrated answer-quality or product gain. The title remains **Reader-Safe Context Action Selection for Multi-Hop Question Answering**. Federated/distributed routing is motivation and diagnostic history, not the paper title or an evaluated systems claim.

## v3 decision rule

The new generator must raise positive-query coverage materially above 20.3% using inference-safe signals only. Coverage below 25% will trigger a `not_ready` main-conference decision; at least 30% is meaningful and at least 40% is strong. Downstream, protocol, breadth, and reproducibility gates remain independent and cannot be waived after seeing test results.
