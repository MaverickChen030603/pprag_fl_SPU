## 7. Computational Cost and Deployment Boundary

| System | Generator ms | Selector ms | Reader ms | Total ms | P95 total | Cross scores | Reader calls | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Top-5 | 0.09 | 0.00 | 140.75 | 140.88 | 252.10 | 0 | 1 | 1.98 |
| Full | 70.05 | 0.61 | 142.59 | 213.48 | 330.56 | 10 | 1 | 2.78 |
| Lite | 6.55 | 0.57 | 136.74 | 143.97 | 254.15 | 0 | 1 | 1.98 |
| Baseline-Truncated-660 | 9.63 | 0.00 | 137.73 | 147.43 | 255.39 | 0 | 1 | 1.90 |
| RECOMP Top1 | 22.04 | 0.00 | 122.65 | 144.76 | 240.11 | 0 | 1 | 2.40 |
| RECOMP 660 | 31.66 | 0.00 | 137.89 | 169.64 | 285.33 | 0 | 1 | 2.40 |

These are measured end-to-end **post-retrieval** times, not reader-only proxies. The shared protocol uses one GPU, batch size one, 50 warmup queries, 500 measured queries, CUDA synchronization, and the same query fingerprint. All online components are recomputed; every final context matches its frozen artifact. Model loading is excluded.

Full's mean is 213.48 ms/query versus 140.88 ms/query for Frozen Top-5, an overhead of 72.60 ms and a 1.52x ratio. Lite lowers the mean to 143.97 ms by removing semantic encoders, but its independent non-inferiority failure prevents promotion to the main method. Every system invokes the answer reader exactly once on the final context. Candidate reader outcomes are offline labels and do not add online reader calls.

Offline work includes action-outcome labeling and fold-specific generator and selector training. Historical offline GPU-hour totals were not recorded and are therefore unavailable. The intended deployment is auditable context organization over a bounded post-retrieval pool. Corpus-scale indexing, dynamic web search, and production streaming are outside the evaluation.
