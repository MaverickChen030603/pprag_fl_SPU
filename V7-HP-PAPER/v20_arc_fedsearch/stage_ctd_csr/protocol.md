# CTD-CSR Protocol

## Scope

This directory is isolated from M0--LR and starts the Context-Teacher
Distilled Client-Specific Retrieval (CTD-CSR) line. This commit authorizes
H0 only: fresh-holdout validation of the already frozen P0 and REM-P candidate
scorers. CT-0 is blocked until the cross-dataset H0 gate passes.

## H0 contract

- Dataset: 2WikiMultiHopQA and MuSiQue.
- Source: calibration records not used by Router-Dev, Router-Calibration,
  R2-A.6 Recovery-Dev, R3 Probe-Dev, or any reader experiment.
- Sampling: the first 300 eligible query IDs ordered by
  `sha256("ctd-csr-h0-v1:" + query_id)`.
- Methods: `P0_single_centroid` and frozen
  `REMP_rrf_p0_dense_lexical` only.
- Candidate cutoffs: 3, 5, and 8.
- Gold: prohibited in inference; used only by the separate offline evaluator.
- Reader/final test: prohibited.

## H0 gate

REM-P proceeds to CT-0 only if both datasets satisfy complete client-set
recall@5 >= P0, at least one improves by >= 0.05 and the other by >= 0.03,
rescue > harm on both datasets, and candidate/ranking/per-query/summary files
are byte-identical across two full runs.

## Artifact policy

Canonical candidate ranking and evaluation files are compared byte-for-byte.
Runtime telemetry is written separately and excluded from comparison. Existing
stage outputs are never overwritten.
