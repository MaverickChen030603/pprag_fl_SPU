# R3-T/R3-C Frozen Hotpot Transfer and Cost-Matched Confirmation

- Reader: not started. Final test: not accessed.
- Hotpot transfer decision: `pass`.
- Cross-dataset reader-gate status: `ready_for_frozen_reader`.

## H0-H5 Retrieval Means
- B0_inherited_route: coverage=0.7400, local=0.5333, raw=0.4700, percentile=0.4500
- B1_static_p0: coverage=0.8033, local=0.5733, raw=0.5100, percentile=0.4900
- B3_label_free_probe: coverage=0.9167, local=0.6600, raw=0.5900, percentile=0.5667
- B4_logistic_seed_20260807: coverage=0.9367, local=0.6767, raw=0.6067, percentile=0.5867
- B4_logistic_seed_20260808: coverage=0.9367, local=0.6767, raw=0.6067, percentile=0.5867
- B4_logistic_seed_20260809: coverage=0.9367, local=0.6767, raw=0.6067, percentile=0.5867

## B4 Logistic vs Inherited Paired Bootstrap
- complete_client_set_recall_at_3: +0.1967, 95% CI [+0.1533, +0.2433].
- local_complete_at_10: +0.1433, 95% CI [+0.1033, +0.1867].
- raw_merged_complete_at_10: +0.1367, 95% CI [+0.0967, +0.1800].

## Cost-Matched Static Baselines
- C0_static_top3: raw=0.5100, total_bytes=7642.6, deep_clients=3, documents=15.
- C1_static_top4: raw=0.5600, total_bytes=10224.4, deep_clients=4, documents=20.
- C2_static_top8_tail_top1: raw=0.5867, total_bytes=10256.7, deep_clients=8, documents=20.

## Local Service Latency
- C0_static_top3: mean=1386.7 ms, median=1369.7 ms; serial local sparse-plus-dense service compute; network latency excluded.
- C1_static_top4: mean=1854.2 ms, median=1844.5 ms; serial local sparse-plus-dense service compute; network latency excluded.
- C2_static_top8_tail_top1: mean=3781.5 ms, median=3783.5 ms; serial local sparse-plus-dense service compute; network latency excluded.
- ProbeRoute_B4_shallow8: mean=2968.6 ms, median=2980.6 ms; serial local sparse-plus-dense service compute; network latency excluded.

## Interpretation
- ProbeRoute is not zero-cost: it queries eight shallow clients and sends 592 B metadata, but retains three deep clients and 15 documents.
- Cost comparison reports actual bytes and local service time. Network transport latency is outside this single-node replay and is not claimed.
- The reader gate is a permission to run the frozen reader protocol, not a reader result.
