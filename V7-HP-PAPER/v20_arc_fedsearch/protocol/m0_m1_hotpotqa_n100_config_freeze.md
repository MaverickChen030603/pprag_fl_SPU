# M0/M1 HotpotQA N=100 Configuration Freeze

**Freeze date:** 2026-08-03
**Status:** retrieval-only configuration selected for independent N=300 replay

## Frozen Contract

| Component | Value |
|---|---|
| Dataset / sample | HotpotQA, V17 frozen development rows 1--100 |
| Client route | inherited V17 topic-silo `selected_clients`, exactly Bc=3 |
| Candidate materialization | all physical client shards only, local depth 10 |
| Formal evaluated clients | only the inherited three selected clients |
| Retriever | frozen BGE + BM25 local hybrid; no LoRA or learned retriever update |
| Local candidate depth | 10 per selected client |
| Transmission budget | 15 documents |
| Global pool | Top-10 |
| Reader | forbidden |

## Selected Simple Configuration

`A1_confidence_proportional + M1_rank_percentile`

- A1 assigns a minimum of two documents per selected client and allocates the
  remaining nine by frozen source confidence, capped at local depth 10.
- M1 ranks each transmitted document by its label-free within-client local-rank
  percentile, breaking ties with frozen dense score and document ID.
- No learned allocator, learned calibrator, gold support, answer text,
  answer presence, reader target, or final-test item is used at inference.

## N=100 Selection Evidence

| Condition | Complete support @10 |
|---|---:|
| A0 equal 5/5/5 + raw | 0.32 |
| A0 equal 5/5/5 + rank percentile | 0.46 |
| A1 confidence proportional + raw | 0.47 |
| A1 confidence proportional + rank percentile | 0.49 |

The two deterministic replays produced byte-identical per-query and summary
matrices.  This freeze is for independent N=300 confirmation only; no reader
metric is used to select it.
