# R3 ProbeRoute-FedRAG: Probe-Dev Final Report

Final state: `query_time_probe_signal_confirmed`. The untouched 100-query Probe-Dev slice was replayed twice per dataset. Exact semantic comparison passed after excluding wall-clock timing fields; no reader was started and final test remains sealed.

## Pre-registered Gate

| Dataset | Best label-free rule | P0 coverage@3 | Best coverage@3 | Delta 95% CI | P0 local@10 | Best local@10 | Rescue/Harm | Gate |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 2wikimultihopqa | P5_static_plus_probe_alpha_0.25 | 0.500 | 0.590 | [+0.020, +0.160] | 0.170 | 0.210 | 12/3 | pass |
| musique | P1_probe_dense_top1 | 0.450 | 0.700 | [+0.160, +0.340] | 0.180 | 0.350 | 26/1 | pass |

## Probe Separability and Offline Ceilings

| Dataset | Static AUPRC | Best probe feature | Probe AUPRC | O0 static Top-3 | O1 oracle subset within Top-8 | O2 CV diagnostic Top-3 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 0.406 | dense_top1_score | 0.764 | 0.500 | 0.810 | 0.490 |
| musique | 0.460 | dense_top1_score | 0.754 | 0.450 | 0.860 | 0.610 |

## Matched 15-document Retrieval and Cost

| Dataset | Best local@10 | Transmitted complete@15 | Raw merged@10 | Percentile merged@10 | Probe bytes | Document bytes | Probe latency ms | Deep latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimultihopqa | 0.210 | 0.210 | 0.210 | 0.210 | 7656 | 7474 | 7366 | 2676 |
| musique | 0.350 | 0.310 | 0.290 | 0.270 | 7695 | 8818 | 5283 | 1979 |

The probe payload contains scalar statistics and bounded title/entity summaries only; it contains neither document text nor embeddings. The offline O2 model is a diagnostic upper bound and is not deployed. The routing signal gate passes on both datasets, but the current verbose probe serialization is not materially smaller than the 15-document payload; therefore a supervised ranker is not yet permitted. The next task is a compact wire-payload audit that leaves routing features and selection rules frozen. Reader evaluation remains prohibited until the separate fresh-holdout gate passes.
