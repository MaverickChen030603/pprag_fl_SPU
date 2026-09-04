# Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## Abstract

Context selection cannot improve a multi-hop question when its candidate action set contains no useful, reader-compatible alternative. A controlled heuristic expansion nearly doubles the number of context actions on HotpotQA yet raises positive-query opportunity only from 20.3% to 23.4%, exposing a candidate-opportunity gap. We address this gap with a fully nested pipeline that first generates bounded context actions using missing-hop estimates, semantic document opportunity, pair complementarity, and answer-anchor-preserving construction, and then applies an action only when a reader-safe selector predicts it to be useful without harming the answer. The generator raises positive-query opportunity to 29.2% and positive-action density from 9.43% to 14.71%, although two of five pre-specified opportunity criteria remain unmet. On a 1,000-query development protocol, official answer F1 improves by 0.0133 (p=0.0176) and supporting-fact F1 by 0.0053 (p=0.0372), while joint F1 shows a positive non-significant trend (+0.0064, p=0.0752). Without further tuning, the frozen pipeline improves answer, supporting-fact, and joint F1 by 0.0088, 0.0056, and 0.0064 on 3,000 disjoint same-source HotpotQA queries; the confirmatory joint result is significant (p=0.0004). Answer and joint directions are consistent for FLAN-T5-Large and UnifiedQA-T5-Large. A frozen 2WikiMultiHopQA transfer yields positive but non-significant answer and joint changes and flat support F1, which bounds rather than establishes cross-dataset generalization. These results show that semantic opportunity creation and risk-controlled selection can convert selective context changes into small, reproducible reader gains.

## 1. Introduction

Multi-hop question answering requires more than retrieving individually relevant passages. The reader must receive complementary evidence, retain an answer-bearing anchor, and encounter facts in a usable order. A context selector can choose only among the alternatives that its generator provides. If none repair missing evidence without harming answer readability, better selection cannot help. We call this the **candidate-opportunity gap**, a concrete source of the broader policy-action-to-reader gap.

Our motivating studies expose the gap. A fixed 4,000-action table provides a positive action for 20.3% of 1,000 HotpotQA queries [@yang-etal-2018-hotpotqa]. Expanding it to 7,882 hand-written actions raises coverage only to 23.4% while positive density stays near 9.4%. We therefore replace template accumulation with semantic, query-conditioned action generation.

V4 estimates missing-hop structure, scores document opportunity and pair complementarity, and constructs at most eight bounded insert, replace, chain, redundancy, and order actions. A separate reader-safe selector predicts answer safety and positive utility, acts within an inner-selected coverage budget, and otherwise preserves the baseline. Generator and selector are trained under five-fold nested query splits; no outer-test answer, support, reader outcome, or oracle action label is available at inference.

The generator raises positive density to 14.71% and query coverage to 29.2%, passing three of five pre-specified criteria. On the 1,000-query development protocol, answer and support F1 improve significantly while joint F1 is positive but non-significant. With the complete pipeline frozen, a disjoint 3,000-query same-source holdout yields significant answer, support, and joint gains. FLAN and UnifiedQA show consistent answer and joint directions, although one support predictor is shared. A frozen 2Wiki transfer [@ho-etal-2020-2wiki] is directionally positive for answer/joint but non-significant, and an official RECOMP reproduction [@xu-etal-2024-recomp] is below V4 under the standardized reader.

Our contributions are: (1) the candidate-opportunity diagnosis; (2) a semantic bounded action generator; (3) fully nested risk-controlled selection; and (4) development and frozen same-source evidence that the combined pipeline converts context opportunity into small, reproducible reader gains.

## 2. Related Work

Multi-hop retrieval composes evidence across retrieval steps [@xiong-etal-2021-mdr], while HotpotQA and 2Wiki expose support annotations [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. Reader-aware selection methods optimize contexts beyond independent relevance [@xin-etal-2025-rcps; @lee-etal-2025-setr; @yu-etal-2024-rankrag]. RECOMP compresses retrieved contexts [@xu-etal-2024-recomp], and context studies show that position and irrelevant passages can alter reader behavior [@liu-etal-2024-lost; @shi-etal-2023-distracted]. V4 differs by separating the candidate-opportunity ceiling from selection, constructing bounded extractive actions over one frozen Top-5 pool, and preserving the baseline through selective fallback [@geifman-elyaniv-2019-selectivenet].

## 3. Problem Setting

For query q, frozen baseline context C0, and fixed reader R, a generator exposes actions A(q). An action is development-positive when it preserves answer F1 and increases answer-evidence utility. Query opportunity is the existence of at least one such action. Any selector over A(q) is upper-bounded by that opportunity. Actions preserve a five-document budget and may insert, replace, chain, remove redundancy, or reorder documents. They never synthesize source text.

The no-leak contract is query-level. Each outer fold trains on 800 queries and acts on 200 disjoint queries. Inner out-of-fold predictions select thresholds and coverage. Target-query answers, support labels, reader outcomes, and oracle action quality are forbidden from generation and selection.

## 4. Semantic Context Action Generation

The missing-hop estimator predicts bridge-missing, answer-resolution-missing, redundant, ordering, or no-intervention states. A document model combines lexical features, MPNet similarity [@song-etal-2020-mpnet], cross-encoder relevance, context similarity, and novelty. A pair model estimates whether two documents form complementary hops. The constructor turns these scores into at most eight actions across six families while protecting inference-time answer anchors and preserving Top-5 budget.

Fully nested ablations show the clearest losses for pair complementarity and two-document actions: removing them reduces overall coverage from 29.2% to 27.7% and 25.1%. Other scoring components are non-monotonic; replacing the document model raises raw coverage but lowers answer safety. We therefore claim value for semantic bounded generation as a whole and for pair/chain structure, not independent necessity of every score.

## 5. Reader-Safe Selection

Two balanced logistic heads predict answer safety and positive opportunity from inference-safe action features. Inner training searches thresholds and coverage from 10% to 30%. Feasible settings limit mean answer loss to 0.001 and selected-action answer-drop rate to 5%. The highest-scoring eligible action is used within the coverage budget; all other queries retain C0. Across outer folds this yields 26.0% development coverage.

## 6. Experimental Setup

We use HotpotQA distractor validation [@yang-etal-2018-hotpotqa]. The 1,000-query development protocol is fully nested. The frozen same-source confirmatory holdout contains 3,000 disjoint queries from the same source ordering and is evaluated after all methods, thresholds, prompts, and the support threshold are fixed. The baseline is HybridSoftRetriever (alpha 0.55, uniform weights, Top-5), not a substituted BM25-only baseline [@robertson-zaragoza-2009-bm25].

The primary reader is FLAN-T5-Large [@raffel-etal-2020-t5]. UnifiedQA-T5-Large [@khashabi-etal-2020-unifiedqa] receives identical contexts as a second answer reader. One support predictor with threshold 0.7 is shared. We report official answer, supporting-fact, and joint EM/F1. Paired confidence intervals and two-sided p-values use 5,000 query bootstrap resamples. Holdout FLAN joint F1 is the confirmatory primary endpoint; answer and support F1 are ordered secondary endpoints.

External validation uses a deterministic 1,000-query sample from 2Wiki development with no target tuning. For comparison, RECOMP uses its official HotpotQA compressor checkpoint, five input documents, and one output sentence, but the reader is standardized to FLAN-T5-Large.

## 7. Results

### 7.1 Opportunity

| Method | Effective actions | Positive-action density | Overall positive-query coverage | Non-ceiling coverage | Newly covered vs predecessor | New-query efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 fixed actions | 4,000 | 9.48% | 20.3% | 32.90% | n/a | n/a |
| V3 heuristic expansion | 7,882 | 9.43% | 23.4% | 38.30% | 81 V2-uncovered queries | 0.0209 |
| V4 semantic generation | 7,934 | 14.71% | 29.2% | 47.63% | 81 V3-uncovered queries | 0.0143 |

V3 adds 3,882 actions relative to V2. V4 exposes 5,655 contexts absent from the V3 table. "Newly covered" is a set difference, not the net coverage change: V3 newly covers 81 V2-negative queries but fails to recover 50 V2-positive queries. V4 passes three of five pre-specified opportunity criteria. Overall coverage (29.2% versus a 30% target) and new-query efficiency do not pass.

V4 covers 292 queries and 47.63% of non-ceiling queries. It passes conditional coverage, marginal breadth, and density criteria but fails the 30% overall-coverage and efficiency criteria. The generator improves opportunity without solving it.

### 7.2 Development and Confirmatory Holdout

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.483 | 0.6114 | 0.073 | 0.4920 | 0.052 | 0.3241 |
| V4 semantic generation + reader-safe selection | 0.493 | 0.6247 | 0.074 | 0.4973 | 0.051 | 0.3305 |
| Delta | +0.0100 | +0.0133 | +0.0010 | +0.0053 | -0.0010 | +0.0064 |

Paired bootstrap, 5,000 resamples: answer F1 [+0.0024, +0.0249], p=0.0176; supporting-fact F1 [+0.0003, +0.0106], p=0.0372; joint F1 [-0.0005, +0.0132], p=0.0752. Answer F1 and supporting-fact F1 improve significantly at the unadjusted 0.05 level. Joint F1 is positive but not significant. The selector intervenes on 260/1,000 queries and its selected-action answer-drop rate is 5.0%.

Development answer and supporting-fact F1 are significant; joint F1 is not. The selector changes 260 contexts and reaches a 5.0% selected answer-drop rate.

| Reader | N | Intervention coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Joint F1 95% CI | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | 3,000 | 0.258 | +0.0088 | +0.0056 | +0.0064 | [+0.0027, +0.0104] | 0.0004 |
| UnifiedQA-T5-Large* | 3,000 | 0.258 | +0.0110 | +0.0056 | +0.0085 | [+0.0049, +0.0122] | <0.0002 |

The 3,000 queries are disjoint from the 1,000-query development sample but come from the same HotpotQA distractor validation source. Generator models, selector thresholds and coverage, reader prompts and decoding, and the support threshold were frozen. FLAN answer F1 p=0.0096; FLAN supporting-fact F1 p=0.0004. *UnifiedQA reuses the frozen selected contexts and the same sentence-support predictor; it is an answer-reader replication, not an independent support-pipeline replication.

On the holdout, FLAN answer, support, and joint gains are significant without retuning. UnifiedQA shows the same answer/joint direction. Because support predictions are shared, this is not independent support replication.

### 7.3 External Baseline and Dataset

| System | Answer F1 | Supporting-fact F1 | Joint F1 | Context protocol |
| --- | ---: | ---: | ---: | --- |
| Frozen Top-5 baseline | 0.6114 | 0.4920 | 0.3241 | Original five documents |
| RECOMP extractive compressor | 0.4437 | 0.3701 | 0.2084 | Official HotpotQA checkpoint, top-1 sentence from Top-5 |
| V4 semantic generator + selector | 0.6247 | 0.4973 | 0.3305 | Bounded five-document context action or fallback |

V4 minus RECOMP: answer F1 +0.1811, supporting-fact F1 +0.1272, joint F1 +0.1221. Classification: `faithful_method_reproduction_with_standardized_reader_adaptation`. We use the official repository at commit `51d4432`, author checkpoint `fangyuan/hotpotqa_extractive_compressor`, and paper settings of five input documents and one selected sentence. The paper's FLAN-UL2 reader is replaced by the frozen V4 FLAN-T5-Large reader to standardize downstream evaluation; this adaptation is stated rather than hidden. Supporting-fact evaluation is an extension that treats the selected sentence as RECOMP's predicted support fact.

RECOMP's one-sentence compression is harmful under this multi-hop reader setting. This is a controlled method comparison, not an exact reproduction of the paper's full reader stack.

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.402 | 0.4709 | 0.080 | 0.4545 | 0.049 | 0.2463 |
| Frozen V4 transfer | 0.407 | 0.4794 | 0.078 | 0.4539 | 0.047 | 0.2496 |
| Delta | +0.0050 | +0.0086 | -0.0020 | -0.0006 | -0.0020 | +0.0033 |

Answer F1: [-0.0021, +0.0191], p=0.1116. Supporting-fact F1: [-0.0036, +0.0025], p=0.6928. Joint F1: [-0.0031, +0.0098], p=0.3296. The HotpotQA generator, selector, thresholds, coverage, reader, and support predictor are frozen; only the data adapter changes. The result is directionally positive for answer and joint F1, statistically flat for support F1, and not significant. It is external validation evidence, not proof of broad cross-dataset generalization. Opportunity density is 14.29%; positive-query coverage is 31.7%; selection coverage is 26.0%; selected-action answer-drop rate is 6.92%.

Frozen 2Wiki transfer is directionally positive for answer and joint F1, statistically flat for support, and non-significant. We treat it as a boundary result rather than a generalization claim.

## 8. Analysis

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

Pair complementarity and two-document construction account for the clearest opportunity losses. The document model and semantic feature removals reveal a breadth-safety trade-off: more raw covered queries can come with lower answer safety and more novel contexts. Opportunity is an upper bound, not a downstream guarantee; only 26% of queries are selected, and the resulting reader gains are smaller than the opportunity change. Development answer-drop rate reaches 5%, falls to 2% on the same-source holdout, and rises to 6.92% on 2Wiki, identifying transfer calibration as the main failure mode.

## 9. Limitations and Ethical Considerations

Two opportunity criteria fail; 70.8% of development queries still lack an observed positive action; and efficiency is low. The 3,000-query result is same-source. External 2Wiki changes are not significant. UnifiedQA shares one support predictor. Component evidence is mixed, only one close external baseline is reproduced, and reader-outcome supervision requires expensive training evaluations. The method cannot recover evidence absent from the local pool. No Federated RAG, privacy, secure aggregation, or SOTA claim is made. Per-query audits and answer-drop reporting are important because selective context changes can hide uneven failures.

## 10. Conclusion

Selection alone cannot cross a missing-candidate ceiling. Semantic bounded action generation raises reader-compatible opportunity, and fully nested risk-controlled selection converts a conservative subset into significant same-source holdout gains. The external result and mixed ablations delimit the next problem: generate broader evidence combinations while calibrating answer safety under distribution shift.
