# V20 M0-Confirm: Cross-Dataset Frozen Retrieval

All rows use the frozen inherited topic route (`Bc=3`), local depth 10, 15 transmitted documents, global top-10, A0 equal allocation, and label-free rank-percentile merge. Reader execution was forbidden and did not occur.

| Dataset | Coverage@3 | Local@5 | Local@10 | A0 raw@10 | A0 percentile@10 | Delta | Rescue | Harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | 0.707 | 0.440 | 0.487 | 0.300 | 0.423 | +0.123 | 37 | 0 |
| 2wikimultihopqa | 0.400 | 0.157 | 0.170 | 0.117 | 0.157 | +0.040 | 12 | 0 |
| musique | 0.423 | 0.137 | 0.190 | 0.120 | 0.123 | +0.003 | 4 | 3 |

## Decision

Final state: `routing_residual_reconfirmed`. audit frozen Bc=3 routing coverage before any reader evaluation.

Reproducibility: both runs were required to be byte-identical for matrix and per-query artifacts.
