# REM-P Evidence Memory Recovery Gate

**Stage:** `REM-P`  
**Status:** new isolated exploration branch  
**Purpose:** test whether a richer client-local representative evidence memory
profile can repair the candidate representation bottleneck observed after
R2-A.5, without changing the frozen federated retrieval contract.

## Motivation

M0/U1/R2-A/R2-A.5 established that:

- HotpotQA has a real cross-client merge calibration signal, but the signal is
  not stable enough across 2WikiMultiHopQA and MuSiQue to serve as the main
  method.
- For 2WikiMultiHopQA and MuSiQue, the largest recoverable loss is upstream:
  support-client routing and candidate representation.
- R2-A resource cards based on single centroids, clustered prototypes, and
  simple lexical sketches do not reliably improve candidate complete-set
  recall.
- R2-A.5 shows both compression opportunity and candidate absence. Therefore a
  learned selector is premature until the client candidate set improves.

REM-P is the narrow recovery gate before any self-distillation or learned
set-selection work. It introduces a richer, still document-local client summary:
representative titles, entity anchors, rare discriminative snippets, and diverse
embedding-selected evidence units.

## Fixed Contract

| Item | Setting |
|---|---|
| Datasets | 2WikiMultiHopQA and MuSiQue only |
| Splits | Router-Dev or a newly frozen Router-Dev slice; no final test access |
| Partition | Existing `topic_silo_m20` partition |
| Client budget | `Bc=3` remains fixed |
| Candidate evaluation | `L in {3,5,8}` clients |
| Local retrieval | Not changed in this gate |
| Reader | Forbidden |
| Training | No router, selector, retriever, reader, calibrator, or distillation training |
| Gold use | Offline metric only after ranked client lists exist |
| Inference features | Query text and client resource profile only |

## Profile Definition

Each client profile is built from local shard documents only. A profile may
contain:

1. `p0_single_centroid`: the baseline client centroid.
2. `representative_units`: bounded evidence memory units, each containing:
   - `unit_id`
   - `unit_type`: `diverse_dense`, `rare_snippet`, or `entity_anchor`
   - `title`
   - short `text`
   - source document id when available
   - selection score and selection reason
   - normalized embedding vector
3. `lexical_memory`: bounded token and entity counters derived only from local
   documents.

The profile is a resource card, not a transferred corpus. It must remain
bounded by `--units-per-client`.

## Methods Compared

The gate compares fixed, non-trained candidate scoring rules:

- `P0_single_centroid`: baseline centroid score.
- `REMP_dense_max`: max query-to-unit embedding similarity.
- `REMP_dense_topk_mean`: mean of top-k query-to-unit similarities.
- `REMP_lexical`: query term overlap with bounded lexical memory.
- `REMP_rrf_dense_lexical`: fixed reciprocal-rank fusion of dense and lexical
  REM-P scores.
- `REMP_rrf_p0_dense_lexical`: fixed reciprocal-rank fusion of P0, REM-P dense,
  and lexical scores.

No method uses labels, support ids, answer strings, reader outputs, or per-query
development tuning.

## Go / No-Go

REM-P passes only if, on an independently frozen development protocol:

1. Both 2WikiMultiHopQA and MuSiQue improve candidate complete-set recall@5 by
   at least `+0.05` absolute over `P0_single_centroid`.
2. Both datasets have rescue > harm at `L=5`, where rescue means P0 incomplete
   and REM-P complete, and harm means P0 complete and REM-P incomplete.
3. Candidate complete-set recall@8 is reported to distinguish representation
   absence from later Bc=3 compression.
4. All outputs are byte-identical across two independent runs.

If the gate fails, the MARS/REM-P method line stops and V20 should be written as
a federated multi-hop retrieval bottleneck audit. If the gate passes, the next
stage may consider a frozen R2-B selector or a separate self-distillation router
proposal, still before reader access.

## Relationship to LEANN, Self-Distillation, and PPML

- LEANN motivates bounded local evidence memory on personal devices, but this
  gate does not require changing the local index implementation.
- Context-based self-distillation is explicitly postponed until the candidate
  representation bottleneck is repaired.
- PPML motivates the privacy-constrained setting; this gate does not claim
  cryptographic privacy.

