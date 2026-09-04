# Full Latency Report

## End-to-end post-retrieval latency

| System | Mean ms/query | Median ms/query | P95 ms/query | Reader calls | CrossEncoder scores | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Top-5 | 140.88 | 124.25 | 252.10 | 1 | 0 | 2.12 GB |
| CrossEncoder-Top5 | 149.90 | 135.47 | 262.59 | 1 | approximately 10 | not separately frozen |
| Full | 213.48 | 199.81 | 330.56 | 1 | 10 | 2.99 GB |

Peak memory uses decimal GB from the recorded byte counters; binary values are approximately 1.98 GiB for Frozen Top-5 and 2.78 GiB for Full.

## Full component means

| Component group | Mean ms/query |
| --- | ---: |
| Generator | 70.05 |
| Selector | 0.61 |
| Serialization plus reader | 142.59 |
| Total | 213.48 |

Semantic feature computation, especially MPNet encoding, dominates the added generator cost. Pair-feature construction and pair classification are small relative to semantic encoding. Full modifies approximately 26% of contexts but executes generator and selector computation for every query.

## Protocol

Measurements use one GPU, batch size one, 50 warmup queries, 500 measured queries, fixed query order, and CUDA synchronization around components. Model loading and upstream retrieval are excluded. Every online system makes one answer-reader call. The numbers describe one post-retrieval implementation and are not throughput, energy, edge-hardware, or production service-level measurements.
