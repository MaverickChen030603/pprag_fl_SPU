# V15 Experiment Status

Updated: 2026-07-21 23:55 JST.

## Executive status

V15 has completed protocol reconstruction, fresh split freezing, real-corpus
indexing, dual-dataset retrieval smoke tests, repair-search smoke tests, and a
100-query HotpotQA dual-reader pilot. No sealed final-test label has been read.
The current status remains `needs_more_experiments`, not `method_breakthrough`.

## Completed

- Audited 1,563 historical structured artifacts. The conservative exclusion
  inventory contains 8,405 HotpotQA IDs, 12,405 HotpotQA normalized questions,
  2,187 2Wiki IDs, and 2,187 2Wiki normalized questions.
- Materialized official train sources: HotpotQA 90,447 queries and 2Wiki
  167,454 queries, with SHA-256 provenance.
- Frozen disjoint train/development/calibration/final splits of
  5,000/1,000/1,000/2,000 for each dataset. Final labels are separated under
  `data/sealed/` and stored mode `0400`.
- Built real train-corpus FTS5 indexes: 481,959 HotpotQA documents and 369,280
  2Wiki documents. Candidate pools use retrieved documents only; there is no
  random or gold-document padding.
- Implemented Top-10 enumerated repair and Top-20 beam repair with exact
  fallback, duplicate pruning, inference-safe scoring, direct multi-reader
  delta prediction, independent empirical gating, and a finite-threshold
  Learn-then-Test scaffold.
- Passed ten local and ten server unit tests. The synchronized server no-leak
  audit also passes with `cascade/` included in its scan.
- Completed 50-query Top-10/Top-20 retrieval smoke tests for both datasets.
- Completed a 100-query HotpotQA pilot with 16 reader-evaluated actions/query
  for both FLAN-T5-Large and UnifiedQA (3,200 reader-action outcomes total).
- Added reproducible per-reader opportunity and cross-reader robust-utility
  diagnostics. Learned action selection uses inference-safe features only;
  reader labels are restricted to retrospective evaluation and oracle rows.
- Implemented and executed the first cheap-gate/cost-aware cascade prototype.
  At the preregistered 0.90 opportunity-recall target, the 78/22-query pilot
  requires threshold 0 and invokes the expensive stage for 100% of queries;
  therefore the cascade has not yet demonstrated a cost saving.

## Retrieval smoke

| Dataset | Pool | Support recall | Complete support | Answer document | Mean retrieval latency |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 10 | 0.870 | 0.740 | 0.900 | 1331 ms |
| HotpotQA | 20 | 0.900 | 0.800 | 0.920 | 1331 ms |
| 2Wiki | 10 | 0.680 | 0.360 | 0.600 | 1139 ms |
| 2Wiki | 20 | 0.710 | 0.400 | 0.660 | 1139 ms |

These are development-smoke diagnostics, not main results. Retrieval latency is
reported separately from post-retrieval action cost.

## HotpotQA 100-query pilot

| Reader | Baseline Answer F1 | Baseline SP F1 | Baseline Joint F1 | Joint-oracle delta | Positive-opportunity queries |
|---|---:|---:|---:|---:|---:|
| FLAN-T5-Large | 0.6553 | 0.4457 | 0.3309 | +0.0528 | 19% |
| UnifiedQA | 0.4955 | 0.4457 | 0.2627 | +0.0474 | 15% |

The same-action robust oracle at beta=0.5 has mean-reader Joint delta +0.0475,
minimum-reader mean delta +0.0185, and zero observed reader harm at 22% oracle
intervention. This is an action-set upper bound, not selector performance.

The MLP direct scorer is stable but not reader-consistent: held-out Joint-delta
Spearman is 0.2184 for FLAN and -0.0670 for UnifiedQA. Per-reader top-action
realized Joint deltas are +0.0068 and -0.0206. A robust beta=1 diagnostic on the
same 22 held-out queries selects one inference-safe action per query and yields
mean-reader Joint delta +0.0113 and minimum-reader mean +0.0022 with no observed
reader harm. The sample is too small for a confirmatory claim.

## Go/No-Go status

- Checkpoint 1, direct utility: `partial_pass`. The action set has real reader
  opportunity and the conservative robust objective has a small positive pilot
  signal, but the direct scorer has not passed cross-reader ranking robustness.
- Checkpoint 2, expanded search: `provisional_pass`. Top-20 exposes 63 non-null
  actions plus exact fallback in 84.6 ms/query, but search-level absence still
  needs full oracle decomposition.
- Checkpoints 3-4: not evaluated.

## In progress / next

1. Increase train-derived reader-labelled queries before increasing actions per
   query; train the objective on robust mean/min utility directly.
2. Evaluate the frozen robust selector on independent development and then
   calibrate per-query gates on calibration only.
3. Complete 2Wiki dual-reader labels; its Top-20 complete-support ceiling is the
   largest current cross-dataset risk.
4. Retrain the cheap gate with substantially more query-level examples. The
   current pilot has 100% expensive-stage invocation and fails the cost target.
   Then measure end-to-end P50/P95, throughput, and GPU memory.
5. Run preregistered baselines and ablations. Keep final-test labels sealed
   until the method, beta, thresholds, and cascade are frozen.
