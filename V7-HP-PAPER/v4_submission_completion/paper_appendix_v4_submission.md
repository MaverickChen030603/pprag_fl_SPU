# Appendix: Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## A. Frozen Configuration

- Development queries: 1,000 HotpotQA distractor-validation examples.
- Outer folds: 5; each fold has 800 training and 200 test queries.
- Generator seed: 20260714.
- Bi-encoder: sentence-transformers/all-mpnet-base-v2.
- Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2.
- Primary reader: google/flan-t5-large, pinned revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a.
- Baseline retriever: HybridSoftRetriever(alpha=0.55, uniform weights, top_k=5).
- Maximum effective actions per query: 8.
- Selector answer-drop risk budget: 5%.
- Support threshold: 0.7.
- Bootstrap resamples: 5,000.

## B. No-Leak Contract

The generator audit reports `1000` outer-test queries, `7934` effective actions, and `5655` contexts absent from V3. Forbidden target-query answer, support, reader-outcome, oracle-action, and post-hoc coverage fields are all false or empty. Each fold records train/test fingerprints and a model hash. The selector audit records five disjoint folds and confirms `outer_test_outcomes_used_for_training_or_tuning: false` for each.

## C. Action Families

| Family | Count | Purpose |
| --- | ---: | --- |
| Single complementary insertion | 2300 | Add one likely missing-hop document while preserving four baseline slots |
| Anchor-preserving replacement | 1132 | Replace a risky/redundant document while retaining an answer anchor |
| Semantic two-document chain | 2387 | Insert a complementary pair for two-hop composition |
| Redundancy replacement | 537 | Exchange a redundant context item for novel evidence |
| Bridge-first reorder | 685 | Move bridge evidence earlier without changing text |
| Answer-anchor-first reorder | 893 | Protect answer readability through order |

## D. Opportunity Definitions and Gates

An action is effective when it differs from fallback. It is answer-safe when answer F1 does not decrease. It is positive when it is answer-safe, improves answer-title product, and either improves title recall or does not reduce title F1. Overall opportunity is the fraction of all queries with at least one positive action. Conditional opportunity excludes 389 diagnostic ceiling queries. New-query efficiency divides newly covered V3-negative queries by contexts absent from V3.

The five criteria are: 30% overall coverage; 45% non-ceiling coverage; at least 70 newly covered V3-negative queries or a seven-point net gain; 12% positive density; and at least 1.25 times V3 efficiency. V4 passes the middle three except efficiency: B, C, and D pass; A and E fail. The criteria are described as pre-specified because no public immutable preregistration record was found.

## E. Full Opportunity Table

| Method | Effective actions | Positive-action density | Overall positive-query coverage | Non-ceiling coverage | Newly covered vs predecessor | New-query efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 fixed actions | 4,000 | 9.48% | 20.3% | 32.90% | n/a | n/a |
| V3 heuristic expansion | 7,882 | 9.43% | 23.4% | 38.30% | 81 V2-uncovered queries | 0.0209 |
| V4 semantic generation | 7,934 | 14.71% | 29.2% | 47.63% | 81 V3-uncovered queries | 0.0143 |

V3 adds 3,882 actions relative to V2. V4 exposes 5,655 contexts absent from the V3 table. "Newly covered" is a set difference, not the net coverage change: V3 newly covers 81 V2-negative queries but fails to recover 50 V2-positive queries. V4 passes three of five pre-specified opportunity criteria. Overall coverage (29.2% versus a 30% target) and new-query efficiency do not pass.

## F. Selector Fold Details

| Fold | Inner-selected coverage | Safety threshold | Positive threshold | Outer selected | Outer answer-drop rate | Outer answer F1 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.30 | 0.6 | 0.3 | 60 | 1.67% | +0.0200 |
| 1 | 0.30 | 0.5 | 0.3 | 60 | 6.67% | +0.0022 |
| 2 | 0.25 | 0.5 | 0.3 | 50 | 10.00% | -0.0029 |
| 3 | 0.15 | 0.6 | 0.3 | 30 | 3.33% | +0.0132 |
| 4 | 0.30 | 0.6 | 0.3 | 60 | 3.33% | +0.0339 |

Fold variation is not hidden: fold 2 has a 10% selected-action answer-drop rate and a small negative answer delta even though aggregate risk is 5%. The nested protocol prevents this fold from changing global thresholds after observation.

## G. Official Development Results

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.483 | 0.6114 | 0.073 | 0.4920 | 0.052 | 0.3241 |
| V4 semantic generation + reader-safe selection | 0.493 | 0.6247 | 0.074 | 0.4973 | 0.051 | 0.3305 |
| Delta | +0.0100 | +0.0133 | +0.0010 | +0.0053 | -0.0010 | +0.0064 |

Paired bootstrap, 5,000 resamples: answer F1 [+0.0024, +0.0249], p=0.0176; supporting-fact F1 [+0.0003, +0.0106], p=0.0372; joint F1 [-0.0005, +0.0132], p=0.0752. Answer F1 and supporting-fact F1 improve significantly at the unadjusted 0.05 level. Joint F1 is positive but not significant. The selector intervenes on 260/1,000 queries and its selected-action answer-drop rate is 5.0%.

## H. Same-Source Confirmatory Holdout

The source is `huggingface:hotpot_qa/distractor/validation`, seed 44. The sample is disjoint from development and the baseline reproduction audit is 1.0. The generator produces 23,724 effective actions. The selector intervenes on 774 queries. No thresholds are retuned.

| Reader | N | Intervention coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Joint F1 95% CI | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | 3,000 | 0.258 | +0.0088 | +0.0056 | +0.0064 | [+0.0027, +0.0104] | 0.0004 |
| UnifiedQA-T5-Large* | 3,000 | 0.258 | +0.0110 | +0.0056 | +0.0085 | [+0.0049, +0.0122] | <0.0002 |

The 3,000 queries are disjoint from the 1,000-query development sample but come from the same HotpotQA distractor validation source. Generator models, selector thresholds and coverage, reader prompts and decoding, and the support threshold were frozen. FLAN answer F1 p=0.0096; FLAN supporting-fact F1 p=0.0004. *UnifiedQA reuses the frozen selected contexts and the same sentence-support predictor; it is an answer-reader replication, not an independent support-pipeline replication.

## I. Multi-Reader Boundary

On development, UnifiedQA answer F1 changes by +0.0129 and joint F1 by +0.0088; answer-drop rate is 1.50%. On the holdout, answer F1 changes by +0.0110 and joint F1 by +0.0085. Sentence-support predictions are shared with FLAN, so support rows are not independent replications.

## J. Generator Component Ablations

| Variant | Effective actions | New actions vs V3 | Positive density | Overall coverage | Non-ceiling coverage | New V3-uncovered queries | Answer-safe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full V4 generator | 7,934 | 5,655 | 14.71% | 29.2% | 47.63% | 81 | 92.66% |
| - missing-hop estimator | 7,952 | 5,619 | 14.47% | 29.0% | 47.30% | 81 | 92.81% |
| - MPNet features | 7,948 | 5,622 | 14.41% | 29.5% | 48.12% | 83 | 92.59% |
| - cross-encoder features | 7,940 | 5,691 | 14.72% | 30.6% | 49.92% | 91 | 92.57% |
| - learned document opportunity model | 7,934 | 6,484 | 14.91% | 32.6% | 53.19% | 110 | 91.74% |
| - pair complementarity | 7,934 | 5,461 | 10.27% | 27.7% | 45.17% | 71 | 93.07% |
| - two-document chain actions | 5,547 | 3,563 | 10.40% | 25.1% | 40.92% | 54 | 93.69% |
| - anchor-preserving families | 5,909 | 4,088 | 16.57% | 27.4% | 44.68% | 73 | 92.45% |
| - redundancy actions | 7,397 | 5,298 | 14.83% | 29.2% | 47.63% | 81 | 92.85% |
| Lexical-only features | 7,952 | 5,652 | 13.87% | 30.7% | 50.25% | 89 | 92.59% |
| Semantic-only features | 7,952 | 5,929 | 14.68% | 30.6% | 49.92% | 97 | 92.48% |

Learned ablations are retrained on each outer-training split and applied to the disjoint outer test fold; structural family removals reuse the frozen fold model. No outcome from the 3,000-query holdout is used. Pair complementarity and two-document actions make the clearest positive contributions. Removing the learned document opportunity model increases raw opportunity coverage to 32.6% but lowers answer safety to 91.74%; lexical-only and semantic-only variants also show that the full generator is not a post-hoc optimum for every opportunity metric. These results support the bounded semantic action space while limiting claims that every scoring submodule is independently necessary. Selector-level V2 diagnostics are reported separately in the appendix because they use a different action table and coverage and therefore are not V4 component ablations.

The full model is fixed before these comparisons. No ablation is selected using the 3,000-query holdout. Removing the document model yields more raw opportunity but lower answer safety; this is a useful negative component result rather than a reason to rewrite the frozen main method.

## K. Historical Selector Diagnostics

V2's 50%-coverage selector diagnostics are retained only as motivation because their action table and coverage differ from V4. Under that protocol, the full selector has answer F1 delta +0.0028 and joint F1 delta +0.0079; removing its nested safety feature changes them to -0.0029 and +0.0062, while removing support features changes them to -0.0023 and +0.0056. These numbers cannot be inserted into the V4 component table as if they were same-protocol ablations. V3 stops before selector evaluation under its frozen continuation rule.

## L. RECOMP Protocol and Results


# Faithful Baseline Protocol

## Selection decision

We considered Reader-Centered Passage Selection, SetR, RECOMP, and RankRAG. RECOMP was selected because an author-maintained implementation and an author-released HotpotQA extractive-compressor checkpoint were directly executable under the frozen V4 evaluation. The SetR repository did not expose a completed evaluation path suitable for this run, and no equivalent executable official package was located for Reader-Centered Passage Selection during the audit. This availability decision was made before inspecting comparison outcomes.

## Reproduction contract

- Official implementation: `https://github.com/carriex/recomp` at commit `51d4432`.
- Author checkpoint: `fangyuan/hotpotqa_extractive_compressor`.
- Paper/code hyperparameters: five input documents and one selected sentence.
- Data: the same frozen 1,000 HotpotQA development queries used by V4.
- Input context: the exact frozen HybridSoftRetriever Top-5 documents.
- Context budget: RECOMP compresses the same Top-5 pool; no alternative BM25 baseline is introduced.
- Reader: the same frozen FLAN-T5-Large reader and prompt used for baseline and V4.
- Tuning: no threshold, checkpoint, prompt, or hyperparameter is tuned on the 3,000-query holdout.
- Metrics: official answer, supporting-fact, and joint EM/F1. Supporting-fact scoring is an explicit extension that treats the selected sentence as the predicted support fact.

## Classification and limitation

The comparison is classified as **faithful method reproduction with standardized reader adaptation**, not an exact end-to-end reproduction of the RECOMP paper. The official compressor, checkpoint, and compression budget are retained, while the original FLAN-UL2 reader is replaced to isolate context construction under the V4 reader. This makes the downstream comparison controlled but narrower than reproducing the original paper's full stack.


| System | Answer F1 | Supporting-fact F1 | Joint F1 | Context protocol |
| --- | ---: | ---: | ---: | --- |
| Frozen Top-5 baseline | 0.6114 | 0.4920 | 0.3241 | Original five documents |
| RECOMP extractive compressor | 0.4437 | 0.3701 | 0.2084 | Official HotpotQA checkpoint, top-1 sentence from Top-5 |
| V4 semantic generator + selector | 0.6247 | 0.4973 | 0.3305 | Bounded five-document context action or fallback |

V4 minus RECOMP: answer F1 +0.1811, supporting-fact F1 +0.1272, joint F1 +0.1221. Classification: `faithful_method_reproduction_with_standardized_reader_adaptation`. We use the official repository at commit `51d4432`, author checkpoint `fangyuan/hotpotqa_extractive_compressor`, and paper settings of five input documents and one selected sentence. The paper's FLAN-UL2 reader is replaced by the frozen V4 FLAN-T5-Large reader to standardize downstream evaluation; this adaptation is stated rather than hidden. Supporting-fact evaluation is an extension that treats the selected sentence as RECOMP's predicted support fact.

RECOMP versus the uncompressed baseline has answer F1 delta -0.1678, support F1 delta -0.1219, and joint F1 delta -0.1157; all paired intervals lie below zero. This result is specific to one-sentence compression under the standardized V4 reader.

## M. Frozen 2Wiki Transfer

The 2Wiki adapter preserves answers, document text, and support labels. The deterministic hash sample has 1,000 queries and is chosen without labels. The baseline remains the same hybrid Top-5 construction. Target-dataset training and tuning are disabled.

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.402 | 0.4709 | 0.080 | 0.4545 | 0.049 | 0.2463 |
| Frozen V4 transfer | 0.407 | 0.4794 | 0.078 | 0.4539 | 0.047 | 0.2496 |
| Delta | +0.0050 | +0.0086 | -0.0020 | -0.0006 | -0.0020 | +0.0033 |

Answer F1: [-0.0021, +0.0191], p=0.1116. Supporting-fact F1: [-0.0036, +0.0025], p=0.6928. Joint F1: [-0.0031, +0.0098], p=0.3296. The HotpotQA generator, selector, thresholds, coverage, reader, and support predictor are frozen; only the data adapter changes. The result is directionally positive for answer and joint F1, statistically flat for support F1, and not significant. It is external validation evidence, not proof of broad cross-dataset generalization. Opportunity density is 14.29%; positive-query coverage is 31.7%; selection coverage is 26.0%; selected-action answer-drop rate is 6.92%.

## N. Reproducibility Artifacts

The completion directory contains scripts `01` through `09`, summary JSON files for external validation, faithful baseline, and generator ablation, the frozen paper tables, `references.bib`, and claim/statistical/reproducibility audits. Large per-query reader outputs remain on the experiment server and are referenced by the audit manifests.
