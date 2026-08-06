# V20 Stage R3: ProbeRoute-FedRAG preregistration

## Frozen contract

- Datasets: 2WikiMultiHopQA and MuSiQue.
- Probe-Dev: the previously unused `router_dev[200:300]` rows for each dataset.
- Candidate generator: inherited P0 single-centroid profile; shallow candidate cutoffs `L=5,8`.
- Probe: each candidate client performs local dense/sparse retrieval to depth 10 and returns scalar scores, ranks, and finite title/entity summaries only. It returns no document text, full passage, dense embedding, answer, support label, or reader result.
- Deep phase: exactly 3 selected clients, local depth 10, 5 transmitted documents per client (15 documents total), global pool 10. The retriever and partition remain frozen.
- Reader and final test are forbidden for all R3 Probe-Dev work.

## Fixed label-free baselines

`P0` static P0 Top-3; `P1` probe dense-top1; `P2` probe dense-top3 mean; `P3` dense rank-percentile; `P4` dense/BM25 client-rank RRF; `P5` min-max static plus normalized dense-top3 mean with static weights `0.25`, `0.50`, and `0.75`.

`O2` is an explicitly offline-only five-fold logistic cross-validation upper bound. It may use support-derived labels after all features are frozen, is never exported as a deployable router, and cannot trigger reader evaluation.

## Gate

Before any supervised ranker, a label-free method must improve complete client-set coverage@3 by at least 5 points on both datasets, or by 8 points on one and 3 points without regression on the other; local-complete@10 must improve by 3 points, rescue must exceed harm, and the two deterministic replays must agree. Otherwise R3 stops at the audit.
