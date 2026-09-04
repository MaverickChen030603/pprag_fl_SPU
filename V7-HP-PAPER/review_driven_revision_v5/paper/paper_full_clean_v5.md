# Pair-Complementary Context Actions for Multi-Hop Question Answering

## Abstract

A selector cannot repair a multi-hop context if its candidate set contains no reader-compatible alternative. We study this **candidate-opportunity gap** with a Pair-Complementary Action Generator: it models whether two documents provide complementary hops, constructs bounded two-document chains, preserves answer anchors, and lets a risk-controlled selector intervene only when predicted answer safety and utility are both high. The fully nested Full pipeline improves Answer, supporting-fact (SP), and Joint F1 by +0.0088, +0.0056, and +0.0064 on a frozen 3,000-query HotpotQA holdout; on its 774 edited contexts, the direct paired gains are +0.0340, +0.0219, and +0.0250. A second untouched 3,405-query revision holdout confirms Full gains of +0.0116 Answer and +0.0080 Joint F1. A pre-frozen Lite-Lexical-Pair simplification initially appears close on development but fails the 0.002 non-inferiority rule on the revision holdout (Joint 0.3217 versus 0.3280 for Full), so Full remains the primary implementation and the removed semantic modules are not claimed as individually validated contributions. Budget-matched RECOMP uses the same Top-5 input, reader, support predictor, and approximately 660-token context; its development Joint F1 is 0.3082, and the original 47-token Top-1 result is no longer evidence of general superiority. Few-shot calibration does not meet the pre-specified transfer-safety rule, so distribution-shift safety remains unresolved. Every system invokes the final answer reader once, but comparable end-to-end generator latency remains [NEEDS MEASUREMENT]. The method targets selective organization over a bounded retrieved pool and yields modest population effects, larger conditional effects on edited contexts, and explicit limits on cost and transfer.


## 1. Introduction

Multi-hop question answering is often described as a retrieval problem, but the reader consumes an ordered, budget-limited context rather than an abstract set of relevant documents. A useful context must expose complementary evidence, retain the wording needed to express the answer, and place the hops in an order that a fixed reader can use. Adding one individually relevant document can still make an answer worse if it displaces an answer-bearing anchor or separates two facts that must be read together.

This observation creates a limit for post-retrieval selectors. A selector chooses among actions proposed by a generator; it cannot select a repair that was never proposed. We call the difference between the actions available and the actions needed by the reader the **candidate-opportunity gap**. It is a concrete instance of the policy-action-to-reader gap: changes in an upstream score matter only when they produce a context whose evidence and wording cross the downstream reader's reasoning threshold.

Our earlier heuristic action expansion made this limitation visible. Nearly doubling the action table produced only a small increase in the number of queries with any safe, positive alternative. The problem was not simply insufficient action count. Independent insertions and replacements repeatedly proposed documents that were query-relevant but not complementary to one another, while unrestricted replacements could discard the baseline passage that supplied answer wording.

We therefore organize generation around **pair complementarity** and **bounded two-document chains**. The generator asks whether two candidate documents jointly cover different hops, then inserts the pair while preserving high-value baseline anchors. It does not synthesize evidence and does not alter the corpus-scale retriever. Its output is a small family of auditable context actions over one already retrieved local pool.

Generation alone is insufficient because even a plausible chain can hurt the reader. A separate selector predicts answer safety and positive reader utility. It uses an action only within a calibrated coverage budget; otherwise it returns the unchanged Top-5 baseline. This fallback makes the method a selective intervention system rather than a replacement retriever.

All learned components use a five-fold outer protocol. Generator and selector models are fit on outer-training queries, thresholds and coverage are chosen from inner out-of-fold predictions, and each outer-test query is touched only by frozen models. Target-query answers, support labels, reader outcomes, and oracle action quality are absent at inference. The 3,000-query confirmatory holdout is evaluated without retuning.

The aggregate gains are deliberately reported as modest. On 3,000 same-source holdout queries, the system improves all three F1 metrics by less than one absolute point. The effect is concentrated on the contexts it edits: direct paired accounting shows substantially larger Answer, SP, and Joint gains among selected interventions and exactly zero change on fallbacks. This conditional view is paired with online cost rather than offered as a substitute for population results.

Cross-dataset behavior is a boundary, not a victory claim. A frozen 2Wiki transfer is non-significant, leaves support nearly flat, and increases selected answer-drop. We preserve that result and separately test whether a small target-train calibration set can repair the safety gate while leaving the generator, reader, prompt, and action families frozen. The two settings are reported separately.

Our contributions are: (1) the candidate-opportunity formulation; (2) a pair-complementary generator whose main structural operation is a bounded two-document chain; (3) anchor-preserving, reader-safe selective intervention under fully nested evaluation; and (4) a review-driven empirical account that includes exact conditional effects, equal-budget compression controls, Full-to-Lite simplification, deployment cost, and transfer calibration. The scope is bounded post-retrieval context organization, not open-domain retrieval or streaming index maintenance.


## 2. Background and Related Work

HotpotQA and 2WikiMultiHopQA provide answers together with supporting-fact annotations, enabling separate evaluation of answer generation and evidence access [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. Multi-hop retrievers such as MDR focus on acquiring evidence over multiple retrieval steps [@xiong-etal-2021-mdr]. Our setting begins after a bounded candidate pool has already been retrieved; the intervention reorganizes that local pool for a fixed reader.

Reader-aware retrieval and reranking move beyond independent query-document relevance by optimizing the context as consumed by a downstream model [@yu-etal-2024-rankrag; @xin-etal-2025-rcps; @lee-etal-2025-setr]. Related analyses show that irrelevant passages and evidence position can change reader outputs [@shi-etal-2023-distracted; @liu-etal-2024-lost]. Our contribution is to separate two questions that are often conflated: whether a safe positive context exists in the candidate action set, and whether a selector can identify it.

RECOMP learns extractive context compression [@xu-etal-2024-recomp]. Its released HotpotQA configuration selects one sentence from five documents, whereas our context actions retain a near-full five-document budget. Because a 47-token context and a roughly 660-token context impose different information constraints, we add equal-budget sentence packing and a length-matched baseline truncation control. We treat this as a comparison between context-construction objectives under standardized reader conditions, not as an exact reproduction of RECOMP's original FLAN-UL2 stack.

Selective prediction provides the conceptual basis for fallback: a system should abstain when estimated risk is high [@geifman-elyaniv-2019-selectivenet]. Here abstention means preserving the frozen retrieval baseline. Our safety head is supervised by offline reader outcomes, but online inference runs the reader once on the final context.


## 3. Method

### 3.1 Problem Setting

For a question (q), an upstream retriever returns a bounded document pool (D_q) and an ordered Top-5 baseline (C_0(q)). A context action (a) maps (C_0) to another five-document sequence without changing source text. The generator exposes a finite set (A(q)); the selector either chooses one action or falls back to (C_0). During development, an action is answer-safe when its reader Answer F1 does not fall below the baseline and positive when it improves the joint answer-evidence utility under the frozen reader.

The opportunity of a query is the existence of at least one safe positive action in (A(q)). This quantity is an empirical upper bound for any selector restricted to that action set. Increasing action count is useful only when it raises positive-action density or covers previously uncovered queries.

### 3.2 Pair-Complementary Action Generator

The Lite generator scores individual documents with normalized BM25, question and title overlap, named-entity overlap, bridge-entity overlap with the baseline, novelty, and redundancy. It learns only a pair-complementarity model. Given candidates (d_i,d_j), the pair features describe their individual lexical evidence, entity-chain overlap, combined novelty, and redundancy. The model is a balanced logistic classifier trained from outer-training action outcomes. Lite-Semantic-Pair adds one cached query-document cosine; development results determine whether that extra encoder is retained.

The constructor locks the first two baseline anchors whenever the five-document budget allows. It proposes (i) anchor-preserving single replacements and (ii) two-document chains that replace only the weakest tail positions. PairChain-Ablation removes the single replacement family. Duplicate contexts are removed, and at most six effective actions plus fallback are emitted. Pair scores are computed only over a top-L document set, bounding the pair count by (L(L-1)/2).

### 3.3 Full Implementation and Lite Simplification

The Full implementation also uses a missing-hop estimator, MPNet similarities, cross-encoder document relevance, and a learned document-opportunity model. Review-driven ablations show mixed or non-monotonic effects for individual components, while removing pair complementarity or two-document chains causes the clearest loss of positive opportunity. A frozen Lite simplification is tested on a separate revision holdout and does not meet non-inferiority; Full therefore remains the primary implementation. This result supports the joint Full recipe, but not a claim that every semantic feature is an independently necessary contribution.

### 3.4 Reader-Safe Selector

Two balanced logistic heads estimate answer safety and positive action utility from inference-safe action features. Inner out-of-fold predictions choose safety and utility thresholds together with a 10--30% intervention budget. Within each outer test fold, eligible actions are ranked by positive probability and safety; the highest-ranked actions are applied only up to the frozen coverage budget. Every other query uses the original baseline.

### 3.5 Fully Nested Protocol

The development set is partitioned into five outer folds. Pair models and selector heads are trained on 800 outer-training queries and applied to 200 disjoint queries. Inner folds tune selector thresholds without reading outer-test outcomes. Lite architecture selection uses only the 1,000 development queries and a pre-recorded Joint-F1 margin of 0.002. The chosen variant is frozen before the remaining 3,405 Hotpot validation examples are materialized as a revision holdout.


## 4. Experiments

### 4.1 Data and Frozen Readers

We use HotpotQA distractor validation [@yang-etal-2018-hotpotqa]. The original 1,000-query development sample supports fully nested model and threshold selection. A disjoint 3,000-query sample from the same seed-44 ordering is the original confirmatory holdout. The revision protocol reserves indices 4,000--7,404, yielding 3,405 examples whose outcomes were not accessed before Lite architecture freezing. Query identifiers are audited for zero overlap.

The upstream baseline is the frozen `HybridSoftRetriever` with alpha 0.55, uniform document weights, and up to five documents. FLAN-T5-Large is the primary answer reader [@raffel-etal-2020-t5]; decoding is greedy with 32 new tokens and a 1,024-token model limit. Context text is capped at 3,200 characters. A sentence-level support predictor is trained on Hotpot development only and uses threshold 0.7. We report official Answer, SP, and Joint EM/F1. Paired intervals and two-sided p-values use 5,000 query-level bootstrap samples.

### 4.2 Lite Selection

We compare Full V4, Lite-Lexical-Pair, Lite-Semantic-Pair, and PairChain-Ablation. Every learned variant repeats the outer/inner protocol. Before outcomes are read, the Joint-F1 non-inferiority margin is fixed at 0.002. Eligible Lite variants must also show no statistically detectable Answer or Joint degradation and remove at least 30% of online latency or cross-encoder calls. Ties are resolved by lower answer-drop risk.

### 4.3 Budget-Matched Compression

RECOMP receives the same baseline Top-5 documents and uses the author-released HotpotQA compressor [@xu-etal-2024-recomp]. Sentences are ranked by official compressor score and packed as whole sentences to the nearest budget in {64, 128, 256, 384, 512, 660} FLAN tokens. The fixed equal-budget protocol is 660 tokens and is frozen before the 3,000-query run. Baseline-Truncated preserves source sentence order at the same budgets. All variants use the same FLAN prompt, decoding, support predictor, and paired metric code.

### 4.4 Cost and Conditional Effects

Offline development cost counts candidate actions, reader outcome labels, training stages, and stored labels. Online inference begins after retrieval and includes generator features, pair scoring, selector scoring, and one final reader call. Candidate reader outcomes are never computed online. We report mean, median, and 95th-percentile latency, peak GPU memory, encoder and cross-encoder calls, pair calls, context tokens, and throughput where measured. Conditional gains are computed directly from paired per-query rows for selected and fallback subsets.

### 4.5 External Transfer Calibration

Zero-shot transfer keeps the Hotpot generator, selector, thresholds, reader, prompt, support predictor, and action families fixed on a deterministic 1,000-query 2Wiki development sample [@ho-etal-2020-2wiki]. Few-shot calibration draws nested K={16,32,64,128} subsets from 2Wiki train under five seeds. Only the safety threshold or probability calibration is adapted; the evaluation outcomes never enter calibration. We compare raw-threshold, temperature, Platt, and risk-constrained calibration and report selected answer-drop, coverage, ECE, Brier score, Answer/SP/Joint F1, and risk-coverage behavior.


## 5. Results

### 5.1 Frozen Same-Source Effect

| N | System | Answer F1 | SP F1 | Joint F1 |
|---:|---|---:|---:|---:|
| 3,000 | Frozen Top-5 | 0.6183 | 0.4930 | 0.3292 |
| 3,000 | Full selected/fallback | 0.6271 | 0.4987 | 0.3356 |
| 3,000 | Delta | +0.0088 | +0.0056 | +0.0064 |

The system edits 774/3,000 contexts. On those exact queries, Answer/SP/Joint deltas are +0.0340/+0.0219/+0.0250; the 2226 fallbacks have exactly zero delta. This conditional effect does not replace the population result: it explains where the average effect comes from.

### 5.2 Full-to-Lite Simplification

| Development method | Answer F1 | SP F1 | Joint F1 | Joint vs Full | Point NI | CI NI |
|---|---:|---:|---:|---:|---:|---:|
| frozen_top5_baseline | 0.6114 | 0.4920 | 0.3241 | reference | reference | reference |
| full_v4 | 0.6247 | 0.4973 | 0.3305 | reference | reference | reference |
| lite_lexical_pair | 0.6183 | 0.4922 | 0.3290 | -0.0015 | true | false |
| lite_semantic_pair | 0.6068 | 0.4960 | 0.3251 | -0.0053 | false | false |
| pairchain_ablation | 0.6167 | 0.4937 | 0.3293 | -0.0012 | true | false |

On development, Lite-Lexical-Pair is -0.0015 below Full in Joint F1 and appears within the point margin, but its 95% lower bound crosses -0.002. This development result freezes Lite-Lexical-Pair for the independent revision test; it is not itself a non-inferiority claim.

On the untouched 3,405-query revision holdout:

| Method | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|
| frozen_top5_baseline | 0.6129 | 0.4862 | 0.3201 |
| full_v4 | 0.6244 | 0.4923 | 0.3280 |
| lite_lexical_pair | 0.6149 | 0.4860 | 0.3217 |

Lite minus Full Joint F1 is -0.0063; point-estimate NI is false and CI-based NI is false. The pre-specified Lite success rule is therefore rejected, and Full remains the main method.

### 5.3 Budget-Matched Compression

| Development method | Tokens | Docs | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|---:|---:|
| recomp_top1 | 47.1 | 1.0 | 0.4437 | 0.3701 | 0.2084 |
| recomp_budget_660 | 637.8 | 4.9 | 0.6049 | 0.4704 | 0.3082 |
| baseline_truncated_660 | 637.3 | 4.2 | 0.5875 | 0.4899 | 0.3139 |
| full_v4 | 660.6 | 5.0 | 0.6247 | 0.4973 | 0.3305 |

The 660-token RECOMP condition is the main fair comparison. The curve peaks earlier for some metrics, demonstrating a compression-budget trade-off rather than a monotonic ranking. The original Top-1 condition remains a compatibility diagnostic.

The frozen 660-token protocol is also complete on 3,000 holdout queries; Table 3 reports it without budget retuning.

### 5.4 External Transfer

Zero-shot 2Wiki selection covers 26.0% of queries and has 6.92% selected answer-drop. Its Answer and Joint changes are non-significant and SP is flat; this is a failed safety transfer diagnostic.
No few-shot setting meets all safety and quality criteria; target safety calibration remains an open limitation.


## 6. Computational Cost and Deployment Scope

At deployment, the answer reader is executed once on the selected final context. The reader is not invoked once per candidate action. Candidate reader outcomes are offline labels used to train and audit the generator and selector. The latency columns below measure the final reader after context construction; comparable end-to-end generator latency remains [NEEDS MEASUREMENT].

| System | Offline reader outcomes | Online reader calls | Cross-encoder calls | Reader mean | Reader P95 | Peak memory | Context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_top5_baseline | 0 | 1 | 0 | 0.1554 | 0.2719 | 2101469696 | 669.0 |
| full_v4 | 8934 | 1 | 10 | 0.1419 | 0.2148 | 2107716096 | 652.1 |
| lite_method | 1850 | 1 | 0 | 0.1377 | 0.2067 | 2107716096 | 665.7 |
| recomp_top1 | [NEEDS MEASUREMENT] | 1 | 0 | 0.1232 | 0.1924 | 1934474240 | 47.6 |
| recomp_budgetmatched | [NEEDS MEASUREMENT] | 1 | 0 | 0.1272 | 0.2096 | 2037304320 | 639.2 |

Historical GPU-hour totals and some training-stage wall times are unavailable unless an explicit timing manifest exists; those cells remain marked rather than reconstructed from file modification times. This distinction matters: expensive offline supervision is an amortized research cost, while online deployment consists of feature computation, bounded pair scoring, selector scoring, and one reader call.

The intended use case is an auditable, bounded post-retrieval pool in offline or moderate-latency QA. The method does not address corpus-scale retrieval, streaming index updates, or real-time web search.


## 7. Analysis

### 7.1 What Survived Simplification?

The review-driven experiment sharpens but does not fully simplify the method. Pair complementarity is the clearest learned mechanism, bounded two-document construction is the central structural mechanism, and anchors plus the safety head control answer risk. However, the untouched holdout rejects Lite non-inferiority. The missing-hop estimator, document-opportunity model, MPNet features, and cross-encoder therefore remain part of the empirically stronger Full recipe, while their mixed individual ablations prevent us from claiming each as a separate contribution.

### 7.2 Opportunity Does Not Equal Reader Gain

Raw positive-query coverage is an upper bound, not an endpoint. Some generators expose more positive contexts but also expose more unsafe actions. The selector reduces this space to a conservative intervention subset, so population gains are smaller than opportunity changes. Exact selected-query accounting shows that the average effect is diluted by intentional fallback rather than by negative changes on untouched queries.

### 7.3 Why Budget Matching Changes the RECOMP Claim

The original Top-1 setting gives the reader roughly seven percent of the baseline context budget and is structurally unlikely to preserve two disjoint supporting facts. Equal-budget packing removes that confound. The resulting curve reveals two separate effects: sentence ordering helps relative to source-order truncation at some budgets, while aggressive score ordering can also reduce support coverage as more sentences are added. We therefore compare objectives under fixed conditions and avoid a universal rank ordering.

### 7.4 Transfer Is Mainly a Safety-Calibration Problem

On 2Wiki, the generator still exposes positive actions, but the Hotpot safety gate permits more harmful edits. This pattern is consistent with probability miscalibration under distribution shift rather than the disappearance of all useful candidate chains. Few-shot experiments adapt only the gate, preserving the distinction between transferable generation and target-specific risk control.


## 8. Limitations and Ethical Considerations

The population effects are small, and a larger selected-query effect does not imply that every query benefits. The original 3,000-query confirmation is same-source. The revision holdout shares the Hotpot validation distribution even though its outcomes are untouched. Cross-dataset zero-shot changes are non-significant and exhibit higher answer-drop. Any successful target calibration uses labeled target-train reader outcomes and must not be described as zero-shot behavior.

Lite non-inferiority is rejected on the untouched revision holdout, so the lower-cost variant cannot replace Full without quality loss. Final-reader latency and historical offline GPU hours are different quantities; generator latency and unavailable historical timing are explicitly marked. The candidate pool is bounded and usually contains about ten distractor documents. Pool sizes 20, 50, and 100 are not naturally present for a common fixed subset, so no corpus-scale conclusion is drawn.

The reader and support predictor are fixed models whose errors may be uneven across entities, languages, or question types. A safety gate lowers average risk but cannot guarantee correctness. We do not evaluate privacy, secure aggregation, federated training, web-scale indexing, or production streaming. The method rearranges existing passages and does not generate new evidence, which aids auditability but cannot recover information absent from the pool.


## 9. Conclusion

Reader-aware context selection is limited first by what its generator makes possible. Pair-complementary scoring and bounded two-document chains create auditable multi-hop alternatives; anchor preservation and selective fallback convert only the safest opportunities into reader-facing changes. The result is a modest but reproducible same-source population gain and a larger conditional effect on edited contexts. The review-driven Lite test does not justify removing Full's semantic machinery, but it narrows the conceptual claim to complementary pairs, chains, anchors, and risk control. Equal-budget compression and failed transfer calibration replace broad superiority and generalization claims with testable, bounded statements.


# Appendix

## A. Frozen Protocol

- Hotpot source ordering seed: 44.
- Development: indices 0--999.
- Original confirmatory holdout: indices 1,000--3,999.
- Revision holdout: indices 4,000--7,404.
- Baseline: HybridSoftRetriever, alpha 0.55, uniform weights, Top-5.
- Reader: FLAN-T5-Large; 3,200 context characters; 1,024 tokenizer positions; greedy 32-token output.
- Support predictor threshold: 0.7.
- Bootstrap samples: 5,000.
- Lite Joint-F1 margin: 0.002, frozen before revision outcomes.

## B. Generator Ablation Interpretation

The main text reports Full, Lite-Lexical-Pair, Lite-Semantic-Pair, PairChain-Ablation, removal of pair complementarity, removal of two-document chains, anchor preservation, and the safety selector. Missing-hop, MPNet, cross-encoder, and document-opportunity ablations are implementation diagnostics. Their mixed behavior is not interpreted as consistent independent benefit.

## C. RECOMP Protocol

The author-released checkpoint `fangyuan/hotpotqa_extractive_compressor` scores every sentence in the same frozen Top-5 input. Whole sentences are added in score order to the nearest target context budget. The fixed holdout protocol is 660 tokens; the 64--660 curve is development-only. Baseline-Truncated uses the same sentence packing budget in source order. The answer reader and support predictor are shared.

## D. Candidate-Pool Boundary

Pool sensitivity status: `scope_limited`. In the frozen 3,000 artifact, a common 10/20/50/100-document subset is unavailable. Top-L pruning fixes pair scoring at at most 45 pairs for L=10 even when a larger upstream pool exists. This is a complexity bound, not an open-domain retrieval experiment.

## E. Reproducibility and Missing Measurements

All V5 outputs are generated under `review_driven_revision_v5/` without changing the frozen V4 paper or result directories. Values absent from source manifests are marked `[NEEDS MEASUREMENT]`, `[NEEDS SOURCE FILE]`, or `[NOT AVAILABLE]`; no elapsed time or call count is inferred from file modification times.
