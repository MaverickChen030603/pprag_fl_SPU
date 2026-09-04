# Appendix: Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## A. Frozen Configuration

- Five outer query folds with 800 training and 200 test queries each.
- At most eight effective actions per query and at most five reader documents.
- Primary reader: FLAN-T5-Large with a pinned model revision.
- Second answer reader: UnifiedQA-T5-Large.
- Support threshold: 0.7, shared and frozen.
- Reader input limit: 3,200 context characters and 1,024 tokenizer positions.
- Decoding: 32 new tokens, greedy, no sampling.
- Paired bootstrap: 5,000 resamples.

## B. No-Leak Protocol

For each outer fold, generator and selector training use only outer-training query outcomes. Inner out-of-fold predictions choose selector thresholds and coverage. Outer-test answers, support labels, reader outcomes, oracle action values, and post-hoc coverage are forbidden. The generator audit contains 7,934 outer-test actions and SHA-256 `b269ab83368c329d80dc446d2e8787c640ba92903f5fb6fcd5b605e39bb9bb1e`. No 3,000-query outcome is used to select a generator ablation.

## C. Opportunity Criteria

The five recorded criteria are 30% overall positive-query coverage, 45% non-ceiling coverage, at least 70 newly covered heuristic-negative queries or a seven-point coverage gain, 12% positive-action density, and an efficiency improvement over the heuristic study. The method passes conditional coverage, marginal breadth, and density, but fails overall coverage and efficiency. No formal public preregistration record was located, so the paper uses "pre-specified" rather than "preregistered."

## D. Full Generator Ablation

| Variant | Effective actions | Positive density | Query coverage | Non-ceiling coverage | New heuristic-negative queries | Answer-safe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 7,934 | 14.71% | 29.2% | 47.63% | 81 | 92.66% |
| without missing hop estimator | 7,952 | 14.47% | 29.0% | 47.30% | 81 | 92.81% |
| without mpnet features | 7,948 | 14.41% | 29.5% | 48.12% | 83 | 92.59% |
| without cross encoder features | 7,940 | 14.72% | 30.6% | 49.92% | 91 | 92.57% |
| without semantic document model | 7,934 | 14.91% | 32.6% | 53.19% | 110 | 91.74% |
| without pair complementarity | 7,934 | 10.27% | 27.7% | 45.17% | 71 | 93.07% |
| without two document actions | 5,547 | 10.40% | 25.1% | 40.92% | 54 | 93.69% |
| without anchor preservation | 5,909 | 16.57% | 27.4% | 44.68% | 73 | 92.45% |
| without redundancy actions | 7,397 | 14.83% | 29.2% | 47.63% | 81 | 92.85% |
| lexical only generator | 7,952 | 13.87% | 30.7% | 50.25% | 89 | 92.59% |
| semantic only generator | 7,952 | 14.68% | 30.6% | 49.92% | 97 | 92.48% |

Pair complementarity and two-document chains have the clearest losses. Other component effects are non-monotonic and are not promoted as independent contributions.

## E. Selector Fold Details

| Fold | Coverage | Safety threshold | Positive threshold | Selected | Answer-drop rate | Answer F1 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.30 | 0.6 | 0.3 | 60 | 1.67% | +0.0200 |
| 1 | 0.30 | 0.5 | 0.3 | 60 | 6.67% | +0.0022 |
| 2 | 0.25 | 0.5 | 0.3 | 50 | 10.00% | -0.0029 |
| 3 | 0.15 | 0.6 | 0.3 | 30 | 3.33% | +0.0132 |
| 4 | 0.30 | 0.6 | 0.3 | 60 | 3.33% | +0.0339 |

## F. Statistical Language Boundary

The 3,000-query pipeline was frozen before evaluation, but no immutable pre-run endpoint hierarchy was found. The paper presents FLAN joint F1 as the headline holdout metric and reports answer and SP F1 alongside it. It does not claim formal ordered testing or familywise control. Development and external p-values are supporting analyses.

## G. Multi-Reader Details

On development, UnifiedQA answer and joint F1 change by +0.0129 and +0.0088. On the same-source holdout they change by +0.0110 and +0.0085. The shared support model means these are not independent support replications.

## H. RECOMP Fairness Details

| Property | RECOMP reproduction | Proposed method / baseline |
| --- | --- | --- |
| Input documents | Same frozen Top-5, mean 4.986 matched docs | Same frozen Top-5 |
| Output unit | Top-1 extracted sentence | At-most-five-document bounded context |
| Mean context tokens | 47.13 | 660.57 selected; 668.18 baseline |
| Compression ratio | 7.35% of baseline | Approximately full budget |
| Reader | Frozen FLAN-T5-Large adaptation | Frozen FLAN-T5-Large |
| Support treatment | Selected sentence is predicted support | Frozen sentence-support predictor |

The original RECOMP reader is FLAN-UL2; this audit uses FLAN-T5-Large to standardize the reader. RECOMP answer/SP/joint F1 are 0.4437/0.3701/0.2084; baseline values are 0.6114/0.4920/0.3241. The large gap is confounded by the 7.35% token ratio. A Top-k or token-matched variant would be a new post-hoc setting requiring additional reader runs, so it is not introduced into the frozen main comparison.

## I. External Transfer Details

The 2Wiki sample is deterministic and label-blind. Answer F1 changes from 0.4709 to 0.4794; SP F1 from 0.4545 to 0.4539; joint F1 from 0.2463 to 0.2496. The selected answer-drop rate is 6.92%. All confidence intervals include zero.

## J. Reproducibility

The artifact package includes fold fingerprints, generator model hashes, action and selector outputs, official per-query metrics, same-source disjointness checks, external data-adapter audits, RECOMP checkpoint metadata, component-ablation outputs, and paired-bootstrap summaries. Relative artifact names are used in the anonymous package; local paths and server identifiers are excluded.
