# Candidate Pool Scope Statement

## Observed pool

The experiment begins after retrieval from the HotpotQA distractor candidate pool, which is approximately ten documents per query. In the frozen 3,000-query holdout:

| Threshold | Queries with at least this many documents |
|---:|---:|
| 10 | 2,973 |
| 20 | 1 |
| 50 | 0 |
| 100 | 0 |

There is no common fixed 20-, 50-, or 100-document subset for a same-protocol scaling claim.

## Complexity boundary

Pair construction over a retained set is quadratic in $L$ before pruning: at most $L(L-1)/2$. The frozen deployment uses $L=10$, so 45 pairs are possible before pruning and ten pairs are actually scored per query. The 213.48 ms/query result is valid for this bounded setting only.

## Paper wording

> The method is evaluated as post-retrieval context organization over the bounded HotpotQA distractor pool. It does not establish corpus-scale or streaming behavior.

Future work may test subquadratic pair proposals, approximate nearest-neighbor pair retrieval, adaptive Top-$L$, and streaming calibration. These are new protocols, not conclusions from V5.
