# Final Frozen Online Cost Report

All systems use the same NVIDIA A100-PCIE-40GB GPU on one host, batch size 1, the same first 550 frozen development queries (50 warmup and 500 measured), CUDA synchronization around every timed block, and one FLAN-T5-large reader call per query. Model loading is excluded. Online generation is recomputed, while each final context is checked against the frozen artifact; all match rates are 100%.

| System | Generator ms | Selector ms | Reader ms | Total mean ms | Total median ms | P95 total ms | Throughput q/s | Peak GPU GiB | Encoders | Cross scores | Pairs | Reader calls | Tokens | Dev Joint delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-5 | 0.09 | 0.00 | 140.75 | 140.88 | 124.25 | 252.10 | 7.10 | 1.98 | 0 | 0 | 0 | 1 | 668.7 | +0.0000 |
| Full | 70.05 | 0.61 | 142.59 | 213.48 | 199.81 | 330.56 | 4.68 | 2.78 | 2 | 10 | 10 | 1 | 662.5 | +0.0064 |
| Lite | 6.55 | 0.57 | 136.74 | 143.97 | 129.69 | 254.15 | 6.95 | 1.98 | 0 | 0 | 10 | 1 | 660.6 | +0.0049 |
| Truncated 660 | 9.63 | 0.00 | 137.73 | 147.43 | 131.24 | 255.39 | 6.78 | 1.90 | 0 | 0 | 0 | 1 | 635.8 | -0.0102 |
| RECOMP Top1 | 22.04 | 0.00 | 122.65 | 144.76 | 135.41 | 240.11 | 6.91 | 2.40 | 1 | 0 | 0 | 1 | 46.9 | -0.1157 |
| RECOMP 660 | 31.66 | 0.00 | 137.89 | 169.64 | 152.91 | 285.33 | 5.89 | 2.40 | 1 | 0 | 0 | 1 | 635.6 | -0.0159 |

## Full Component Breakdown

| Component | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|
| document preprocessing | 0.080 | 0.084 | 0.102 |
| lexical features | 2.600 | 2.598 | 3.506 |
| mpnet encoding | 52.867 | 50.072 | 85.833 |
| cross encoder scoring | 11.235 | 10.734 | 16.398 |
| pair feature construction | 1.773 | 1.721 | 2.584 |
| missing hop prediction | 0.484 | 0.478 | 0.536 |
| document opportunity scoring | 0.301 | 0.301 | 0.319 |
| pair complementarity scoring | 0.292 | 0.290 | 0.313 |
| action construction | 0.395 | 0.391 | 0.441 |
| safety head | 0.320 | 0.321 | 0.336 |
| positive utility head | 0.293 | 0.295 | 0.307 |
| final context serialization | 0.019 | 0.018 | 0.021 |
| final reader | 142.593 | 125.286 | 251.423 |

## Interpretation

Full adds +72.60 ms/query relative to Frozen Top-5 under this post-retrieval benchmark. Lite removes the semantic encoders and saves 69.52 ms/query relative to Full, but it failed the independent 0.002 Joint-F1 non-inferiority test and therefore is not the main method.
The reader is not evaluated once per candidate action: every system invokes it exactly once after final context construction. This fact alone is not an efficiency claim; the table reports total online latency.
Offline costs include action-outcome labeling and model training. Historical offline GPU-hour totals were not recorded and are therefore unavailable.
The deployment claim is limited to selective context construction over a bounded post-retrieval candidate pool; open-domain indexing and streaming scalability were not evaluated.
