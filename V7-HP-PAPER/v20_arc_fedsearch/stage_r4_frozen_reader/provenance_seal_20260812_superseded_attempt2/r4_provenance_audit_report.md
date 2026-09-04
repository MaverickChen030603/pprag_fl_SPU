# V20 R4 Provenance Audit Report

## Final Classification

**C — `R4_VALID_EXPLORATORY_ONLY`**

The 7,200-row output is complete, internally consistent, context-aligned, and statistically reproducible. It is not audit-grade confirmatory R4 because the actual root runner accessed labels before/during generation and mixed prediction with evaluation, and because the resumed three-dataset execution did not honor the later Hotpot-only pre-outcome operational contract.

## Artifact Integrity

- Predictions: 7200/7200 rows; 24/24 cells; 300 rows per cell.
- Primary keys: 7200 unique; duplicate keys: 0.
- Input-hash reconstruction mismatches: 0.
- Output-hash mismatches: 0.
- Context alignment mismatches: 0.
- Wholesale cross-method prediction copies detected: False.
- Formal resume cache rows: 0 for both readers, inferred exactly from the first `completed=4` progress record with batch size 4.

## Generation and Labels

The contexts themselves are clean: no gold-answer/support fields were found, and every stored reader input hash reconstructs from question plus context. Nevertheless, `run_r4_reader.py` loads label-bearing manifests before generation, resolves `gold_row` before decoding each batch, computes official metrics immediately, and appends gold, prediction, and metrics together. Thus content leakage into the prompt was not detected, but the strict label firewall and prediction seal were not satisfied.

## Scope and Timeline

The original three-dataset protocol and runner predate outcomes. A later 16:21 JST directive narrowed confirmatory R4 to HotpotQA and stopped the old central jobs. Those jobs resumed at an unknown time/by an unresolved actor and completed. U1 began after the final root predictions and produced no formal reader output; it cannot explain or legitimize the earlier 7,200 rows.

## Statistics

The independent recomputation matches all 24 main rows and 54 paired-bootstrap rows at 1e-12. Logistic ProbeRoute versus federated baseline Joint F1 is positive in 6/6 observed cells. Macro deltas are Answer F1 +0.037664735591, SP F1 +0.075071428571, and Joint F1 +0.038666748670.

## R5 Decision

`R5_PREREGISTRATION_READY = false`. The existing R5 recommendation is a short status artifact, not a complete preregistration, and it relies on a stronger R4 claim than this audit permits. Before R5, the project must explicitly choose either exploratory acceptance or a new sealed confirmatory R4 split. No R5 final test should be opened to resolve this provenance question.
