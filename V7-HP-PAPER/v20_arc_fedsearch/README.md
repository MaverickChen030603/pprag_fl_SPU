# V20 ARC-FedSearch

## Objective

ARC-FedSearch studies query-time federated search with data-local corpora and
matched communication budgets.  It replaces the V17--V19 hypothesis that a
learned context/composer or retriever adapter should create downstream utility.
The question is instead whether adaptive source routing, source-aware document
allocation, calibrated merging, and conditional second-round search recover
more multi-hop evidence than a fixed `Bc=3, local-k=5` federated baseline.

The centralized index is a retrieval reference, not the primary competitor.
It is not assumed to be an oracle evidence upper bound: that designation is
reserved for an explicit gold-only offline oracle-local audit.
The primary comparison is a fixed-budget federated retrieval contract with the
same mean client contacts and transmitted documents.

## Current Stage

`U0_inherited_replay` is an offline loss decomposition over frozen V17
development pools.  It is intentionally retrieval-only.  It does not start a
reader and does not open final-test assets.  Its output chooses the next
implementation layer; it cannot itself establish a deployable improvement.

## Decisions inherited from V17--V19

- Do not continue generic cross-client context composition or reader-aligned
  selective parameter upload as a main method.
- Freeze BGE for the initial routing/merging study.  Do not use LoRA movement
  as a substitute for evidence coverage.
- Keep raw document corpora and indexes client-local.  This is a data-local,
  access-controlled setting; no formal privacy claim is made.
- Gold support, answer text, and support-client IDs are audit/evaluation-only.

## Directory Contract

`stage0_loss_audit/` contains the only executable work before router training.
Later modules must be enabled only after its go/no-go report identifies a
recoverable bottleneck.
