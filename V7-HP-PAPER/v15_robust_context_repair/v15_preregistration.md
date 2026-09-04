# V15 Preregistration

Status: protocol freeze in progress; no V15 outcome-bearing final-test run has
been inspected.

## Confirmatory scope

- Datasets: HotpotQA distractor train-derived frozen test and
  2WikiMultiHopQA train-derived frozen test.
- Previously inspected HotpotQA validation queries (all 7,405) are excluded
  from any claim of fresh confirmation. Previously used train IDs discovered by
  `01_protocol_and_data_audit.py` are also excluded.
- Splits per dataset: train 5,000; development 1,000; calibration 1,000; final
  frozen test 2,000. The deterministic seed is `20260721`.
- Final-test labels are stored separately under `data/sealed/`; method and
  threshold development may use train/development/calibration only.

## Frozen retrieval contract

- Real pools: Top-10 and Top-20. Top-50 is exploratory and may only be reported
  separately if the retriever actually returns 50 unique documents.
- Retriever family: the inherited hybrid dense/sparse retriever, with alpha
  fixed to `0.55`, uniform document weights, and no label-derived features.
- Dense encoder: `BAAI/bge-base-en-v1.5`.
- Sparse scorer: BM25 over title plus paragraph text.
- Reranker baseline: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Final reader budget: five documents. Duplicate normalized titles are removed.
- Candidate pools may contain only retriever outputs; no random or gold padding.

## Readers

- Reader A: `google/flan-t5-large`.
- Reader B: `allenai/unifiedqa-v2-t5-large-1363200`.
- Reader checkpoints, prompts, decoding, maximum input length, and evaluation
  normalization are frozen before development results are inspected.
- Reader-specific labels are retained. Primary robust utility is mean minus
  `beta * standard deviation`; beta is selected once on development from
  `{0, 0.25, 0.5, 1.0}`.

## Repair search

- For pool size `L <= 12`, enumerate all five-document subsets. For each subset
  generate baseline-preserving, retrieval-score, CrossEncoder, and bridge-first
  orders when distinct. Retain Top-K by an inference-safe cheap score, with
  `K in {8, 16, 32, 64}` selected on development.
- For `L > 12`, start from Top-5 and run replace, insert-and-remove, and reorder
  edits. Beam width is in `{8, 16, 32}` and depth in `{1, 2, 3}`. The exact
  baseline is always the null action.
- Duplicate sequences are removed by ordered document IDs. No target-query
  outcome or support label is available to generation, scoring, or gating.

## Action scorer

Primary first-stage model: GBDT or MLP direct multi-task scorer. It predicts,
for each frozen reader, Answer F1 delta, SP F1 delta, Joint F1 delta,
`P(Answer drop)`, and `P(Joint drop)`. Loss terms are regression, harm
classification, and within-query pairwise ranking. DeepSets, Set Transformer,
or an action CrossEncoder may be attempted only if the simple scorer fails the
first Go/No-Go checkpoint.

## Gates and risk

- The deployed decision is independent per query; the V14 known-batch global
  Top-B policy is not the V15 main method.
- Empirical gate: apply the highest-utility action only when predicted utility
  exceeds `tau_U` and predicted Answer harm is below `tau_H`; otherwise return
  the exact baseline.
- Risk-calibrated gate: thresholds are selected only on the independent
  calibration split for target selected-harm rates `{0.04, 0.05, 0.08}`.
- A finite-sample or distribution-free guarantee may be claimed only after the
  assumptions and implementation pass `formal_assumption_audit.md`. Otherwise
  the method is called an empirically calibrated per-query risk gate.

## Cost contract

- A cheap opportunity gate uses BM25, rank, lexical/entity overlap, cached
  dense similarity, and frozen baseline confidence.
- Expensive repair/scoring is invoked only for eligible or uncertain queries.
- Primary target: mean post-retrieval latency <=170 ms/query, or >=20% lower
  than V14 Full, with P50/P95, throughput, memory, and invocation rate reported.

## Primary methods and metrics

Methods: Frozen Top-5, CrossEncoder-Top5, V14 Full, RECOMP, source-order
truncation, marginal utility, direct-delta MLP, fixed-pool subset utility, and
V15 repair. Primary metrics are Answer F1, SP F1, Joint F1, selected Answer-drop
rate, selected Joint-drop rate, coverage, mean/P95 latency, and throughput.
Answer/SP/Joint EM are secondary.

Statistics: query-paired bootstrap with 5,000 resamples, 95% confidence
intervals, paired p-values, effect sizes, and Benjamini-Hochberg correction for
secondary comparisons. Seeds: `20260721`, `20260722`, `20260723`.

## Fixed ablations

1. old structured generator vs expanded repair;
2. title proxy vs direct Joint delta;
3. binary heads vs multi-task scorer;
4. single-reader vs robust multi-reader;
5. batch Top-B vs per-query gate;
6. empirical vs formally audited risk gate;
7. no cascade vs cost-aware cascade;
8. Top-10 vs Top-20;
9. beam width/depth;
10. exact fallback removal.

## Decision checkpoints

- 2026-08-10: direct-delta ranking exceeds the old proxy and development is no
  worse than V14 Full; otherwise keep GBDT/MLP and stop neural escalation.
- 2026-08-25: expanded search reduces search-level absence without uncontrolled
  latency; otherwise retain only enumerated Top-K.
- 2026-09-10: at least two preregistered success conditions must hold.
- 2026-09-20: a method-paper recommendation requires two datasets, two readers,
  complete risk-coverage and cost analysis, and a non-dominated main method.

