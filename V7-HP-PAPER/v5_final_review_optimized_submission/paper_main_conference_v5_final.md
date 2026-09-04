# Pair-Complementary Context Construction with Reader-Safe Selection for Multi-Hop QA

## Abstract

Multi-hop question answering depends not only on retrieving relevant passages but also on constructing a context that exposes complementary evidence to a fixed reader. We study the **candidate-opportunity gap**: a selector cannot recover a reader-compatible context when its candidate actions omit the needed evidence combination. Our Full method scores pair complementarity, constructs bounded two-document chains, preserves answer-bearing anchors, and applies an action only through a fully nested two-head reader-safe selector. On a frozen 3,000-query HotpotQA holdout, Full improves Answer, supporting-fact (SP), and Joint F1 by +0.0088, +0.0056, and +0.0064. An untouched 3,405-query same-source holdout confirms gains of +0.0116, +0.0061, and +0.0080. On policy-selected interventions, the corresponding descriptive gains are +0.0340/+0.0219/+0.0250 on the first holdout; these conditional values are not causal effects and accompany, rather than replace, the modest population results. A frozen Lite simplification fails a 0.002 Joint-F1 non-inferiority criterion, so Full remains primary. Under an approximately matched 660-token budget, adapted RECOMP sentence packing does not improve the frozen baseline. Measured post-retrieval inference is 213.5 ms/query for Full versus 140.9 ms/query for the baseline, with one final reader call in both. Finally, 2Wiki calibration lowers selected answer-drop from 6.92% to 5.10% but misses the pre-specified 4% target. The evidence supports bounded same-source context construction, not broad efficiency or transfer claims.

## 1. Introduction

Multi-hop question answering (QA) is often framed as finding relevant documents, but a reader consumes an ordered and budget-limited context rather than an unordered relevance set. To answer correctly, that context must expose complementary hops, retain the passage that gives the answer its lexical form, and place the evidence where the reader can use it. More retrieval is therefore not automatically better: adding an individually relevant document can displace an answer anchor or leave two necessary facts disconnected.

This creates a structural limit for post-retrieval selection. A selector can choose only among the contexts proposed by its action generator. If no proposal contains a reader-compatible repair, improved selection cannot help. We call this mismatch the **candidate-opportunity gap**. It is observable as a low density of actions that improve reader outcomes without damaging answer quality.

Fixed expansion templates do not reliably close this gap. Independent insertions and replacements often select documents that are relevant to the question but redundant with each other. Unrestricted replacement can also remove the passage that supplies the answer wording. In our frozen development analyses, increasing action count alone yielded little additional safe positive-query coverage.

We instead organize context construction around **pair complementarity**. The Full Pair-Complementary Action Generator models whether two documents supply different parts of a multi-hop chain, constructs bounded two-document actions, and preserves high-value baseline anchors. Its implementation combines lexical, MPNet, and cross-encoder signals with missing-hop, document-opportunity, and pair-complementarity models. These modules form the empirically stronger Full recipe; our conceptual claim is narrower than a claim that each feature helps monotonically.

A separate reader-safe selector decides whether any generated action should reach the reader. One head predicts answer safety and another predicts positive utility. Frozen thresholds and a coverage budget allow selective intervention; otherwise the system returns the original Top-5 context exactly. The online reader runs once, after this decision, rather than once per candidate action.

We use a fully nested five-fold protocol. Generator and selector training occurs on outer-training queries, thresholds are set using inner out-of-fold predictions, and each outer-test query is processed only by frozen components. We then evaluate the frozen Full system on two disjoint same-source holdouts: 3,000 original holdout queries and an untouched 3,405-query revision holdout. Full improves Answer, SP, and Joint F1 on both. The absolute population changes are modest, while direct paired accounting shows larger descriptive changes on the 25-26% of queries where the policy intervenes and exactly zero change on fallbacks.

The revision experiments also define what we do not claim. A development-frozen Lite-Lexical-Pair simplification fails the independent non-inferiority test, so it does not replace Full. A budget-matched compression comparison is reported as a difference between context-construction objectives, not as a universal rank ordering. Frozen transfer to 2Wiki is non-significant and its few-shot gate calibration misses the safety target. Finally, Full is 1.52x the measured baseline post-retrieval latency, so the result is a quality-risk trade-off over a bounded candidate pool rather than an unqualified deployment claim.

Our contributions are:

1. We formulate the candidate-opportunity gap for reader-aware multi-hop context construction.
2. We introduce pair-complementary, anchor-preserving action generation with bounded two-document chains.
3. We combine the generator with fully nested reader-safe selective intervention and exact fallback.
4. We provide two frozen same-source confirmations plus conditional-effect, non-inferiority, budget-matched compression, online cost, and transfer-boundary analyses.

## 2. Related Work

**Multi-hop retrieval and QA.** HotpotQA and 2WikiMultiHopQA pair answers with supporting facts, making it possible to distinguish answer generation from evidence access [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. Multi-step retrievers such as MDR acquire evidence across retrieval steps [@xiong-etal-2021-mdr]. Our setting starts later: a frozen retriever has already produced a bounded candidate pool, and the method reorganizes that pool for a fixed reader.

**Reader-aware context construction.** Reader-aware retrieval and reranking account for downstream behavior beyond independent query-document relevance [@yu-etal-2024-rankrag; @xin-etal-2025-rcps; @lee-etal-2025-setr]. Prior analyses show that distractors and evidence position can alter reader output [@shi-etal-2023-distracted; @liu-etal-2024-lost]. We distinguish action opportunity from action selection: a reader-safe selector remains limited by the combinations exposed by its generator.

**Compression and selective prediction.** RECOMP learns extractive context compression [@xu-etal-2024-recomp]. Because its released Hotpot configuration and our near-full contexts have different budgets and objectives, we use the author-released compressor under a common FLAN reader and include a 660-token condition plus a source-order truncation control. Selective prediction motivates fallback when estimated risk is high [@geifman-elyaniv-2019-selectivenet]. Here fallback means preserving the frozen retrieval baseline, and offline reader outcomes supervise the gate without adding online candidate-reader calls.

## 3. Method

### 3.1 Problem and Opportunity

For a question $q$, a frozen retriever returns a bounded document pool $D_q$ and ordered Top-5 baseline $C_0(q)$. A context action maps $C_0$ to another five-document sequence using only documents in $D_q$ and without editing source text. The generator exposes a finite action set $A(q)$; the selector either chooses one action or returns $C_0$.

During training only, an action is answer-safe when the frozen reader's Answer F1 is no lower than on $C_0$ and positive when it improves the reader-side joint answer-evidence utility. Query opportunity is the existence of at least one safe positive action in $A(q)$. It is an empirical upper bound for any selector restricted to that set, not an inference-time label.

### 3.2 Full Pair-Complementary Action Generator

#### 3.2.1 Candidate document signals

The Full generator computes normalized BM25, question-title and question-text overlap, named-entity overlap, bridge overlap with baseline documents, novelty, and redundancy. It augments these lexical signals with cached MPNet query-document similarities [@song-etal-2020-mpnet] and cross-encoder relevance. At test time these features use only the question, candidate text, baseline ordering, and learned parameters; answers, support annotations, and reader outcomes are absent.

#### 3.2.2 Missing-hop and document-opportunity modules

A missing-hop estimator summarizes which query and baseline signals remain weakly represented. A document-opportunity model then scores whether a candidate can fill that estimated gap while adding nonredundant information. Both models are trained inside each outer fold from training-query outcomes. They are components of the empirically stronger Full implementation, not independently established monotonic contributions.

#### 3.2.3 Pair complementarity

Individual relevance cannot determine whether two documents jointly supply different hops. For each pair among the top candidate set, the generator constructs features from their individual scores, entity-chain overlap, combined novelty, redundancy, and relation to the missing-hop state. A balanced pair classifier estimates complementarity. With at most $L$ candidate documents, pair scoring is bounded by $L(L-1)/2$; the frozen deployment uses ten pair scores per query.

#### 3.2.4 Bounded two-document chain construction

High-scoring complementary pairs form bounded two-document actions. The pair is inserted into weak tail positions of the five-document baseline, producing a compact chain rather than an unconstrained search over permutations. The generator also retains a small single-complementary-insertion family. Duplicate contexts are removed, and the candidate action count is capped before selection.

#### 3.2.5 Anchor preservation and action pruning

The constructor protects the strongest early baseline anchors whenever the budget permits. This prevents an apparently useful support insertion from deleting the passage that supplies answer wording. Actions that duplicate a context, violate the five-document budget, or rank below the frozen pruning rule are discarded. Fallback is always present.

### 3.3 Reader-Safe Selector

The selector has two balanced logistic heads. The safety head estimates whether an action preserves baseline answer quality; the utility head estimates whether it yields a positive reader-side change. Inner out-of-fold predictions determine both thresholds and a 10-30% coverage budget. On an outer-test query, an action is eligible only if it passes both heads. The highest-ranked eligible action is applied while the fold-level budget remains; all other queries receive the exact baseline. The reader then evaluates one final context.

### 3.4 Fully Nested Training and Evaluation

Five outer folds separate training from evaluation. Generator modules and selector heads fit only outer-training queries. Inner folds tune thresholds and coverage without reading outer-test outcomes. Each outer-test query is processed by fold-specific frozen models. The 3,000-query and 3,405-query holdouts are disjoint from the 1,000 development queries, and no holdout outcome selects an architecture or threshold.

### 3.5 Review-Driven Lite Simplification

Lite-Lexical-Pair removes MPNet, cross-encoder, missing-hop, and document-opportunity computation while retaining lexical pair complementarity, bounded chains, anchors, and the two-head selector. Its architecture is selected on development and frozen before the revision holdout is opened. The pre-specified Joint-F1 non-inferiority margin is 0.002. Lite fails this independent test, so Full remains the primary method. The experiment narrows the conceptual explanation but does not show that Full's removed semantic modules are dispensable.

## 4. Experimental Setup

We use HotpotQA distractor validation [@yang-etal-2018-hotpotqa]. A fixed 1,000-query development slice supports nested training and threshold selection. The next disjoint 3,000 queries form the original confirmatory holdout; the remaining 3,405 form an untouched revision holdout. All retain the same source distribution and are not external-domain tests.

The frozen upstream baseline is HybridSoftRetriever with alpha 0.55, uniform document weights, and Top-5 output. FLAN-T5-Large is the primary answer reader [@raffel-etal-2020-t5], using greedy decoding, at most 32 generated tokens, a 1,024-token input limit, and context capped at 3,200 characters. A Hotpot-development support predictor uses a frozen 0.7 threshold. We report official Answer, supporting-fact (SP), and Joint EM/F1. Paired 95% intervals and two-sided p-values use 5,000 query-level bootstrap resamples.

For RECOMP, the author-released HotpotQA compressor scores sentences in the same Top-5 input [@xu-etal-2024-recomp]. Development budgets are 64, 128, 256, 384, 512, and 660 FLAN tokens; 660 is frozen before holdout evaluation. Baseline-Truncated retains source sentence order at the same budget. All systems share reader, prompt, support predictor, and metric code.

For online cost, all systems run on one GPU with batch size one over the same ordered queries. We use 50 warmup queries and measure the next 500, synchronizing CUDA around every component. Model loading is excluded. Online features/actions are recomputed and their final context must exactly match the frozen artifact. Candidate outcome labeling and training are offline.

## 5. Main Results

### 5.1 Two Frozen Same-Source Holdouts

| Split | N | Coverage | Baseline A-F1 | Full A-F1 | A delta (95% CI; p) | SP delta (95% CI; p) | Joint delta (95% CI; p) | Selected A-drop |
|---|---:|---:|---:|---:|---|---|---|---:|
| Original frozen holdout | 3000 | 25.8% | 0.6183 | 0.6271 | +0.0088 ([+0.0023, +0.0152]; 0.0096) | +0.0056 ([+0.0031, +0.0083]; 0.0004) | +0.0064 ([+0.0027, +0.0104]; 0.0004) | 7.75% |
| Untouched revision holdout | 3405 | 25.9% | 0.6129 | 0.6244 | +0.0116 ([+0.0052, +0.0178]; <.0002) | +0.0061 ([+0.0036, +0.0088]; <.0002) | +0.0080 ([+0.0044, +0.0116]; <.0002) | 7.83% |

Full improves all three F1 measures on both holdouts. On the original 3,000-query holdout, the paired deltas are +0.0088 Answer, +0.0056 SP, and +0.0064 Joint F1. The untouched 3,405-query holdout confirms +0.0116, +0.0061, and +0.0080. The latter simultaneously serves as the independent Lite non-inferiority test and a replication of Full. Both sets are disjoint from development, Full was frozen before both runs, and revision outcomes were unread when the Lite architecture was fixed. Because both are HotpotQA same-source samples, we do not pool them for a new significance claim.

### 5.2 Descriptive Effects on Policy-Selected Interventions

Population and conditional views answer different questions. In the original holdout, Full edits 774/3000 contexts (25.8%); Answer/SP/Joint population deltas are +0.0088/+0.0056/+0.0064. Conditional on these policy-selected interventions, the descriptive means are +0.0340/+0.0219/+0.0250. Answer has 89 wins, 60 losses, and 625 ties; Joint has 141/115/518. The selected Answer- and Joint-drop rates are 7.75% and 14.86%. Medians and both interquartile endpoints are zero because most selected contexts tie the baseline.

The revision holdout shows the same concentration pattern: 881/3405 interventions, with descriptive selected Answer/SP/Joint means of +0.0447/+0.0237/+0.0309. These values are descriptive gains conditional on policy-selected interventions. They are not causal treatment effects, expected gains for arbitrary queries, or effects on all improvable queries. In both holdouts, fallback contexts and metrics are exactly unchanged.

### 5.3 Full-to-Lite Non-Inferiority

| Revision-holdout system | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|
| Frozen Top-5 | 0.6129 | 0.4862 | 0.3201 |
| Full | 0.6244 | 0.4923 | 0.3280 |
| Lite-Lexical-Pair | 0.6149 | 0.4860 | 0.3217 |

Lite minus Full Joint F1 is -0.0063 (95% CI [-0.0104, -0.0023], p=0.0004). It misses both the point and interval versions of the frozen 0.002 margin. Lite reduces computation, but the independent quality criterion fails; it is therefore a simplification diagnostic rather than a replacement for Full.

## 6. Budget-Matched Compression

| System (3,000 holdout) | Tokens | Documents | Answer F1 | SP F1 | Joint F1 | E2E ms/query |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Top-5 | 664.5 | 4.986 | 0.6183 | 0.4930 | 0.3292 | 140.88 |
| Baseline-Truncated-660 | 635.7 | 4.236 | 0.6038 | 0.4904 | 0.3224 | 147.43 |
| RECOMP-660 | 635.9 | 4.873 | 0.6226 | 0.4837 | 0.3259 | 169.64 |
| Full | 656.1 | 4.986 | 0.6271 | 0.4987 | 0.3356 | 213.48 |

RECOMP-660 changes Answer/SP/Joint F1 relative to Frozen Top-5 by +0.0043/-0.0093/-0.0033. The Joint interval is [-0.0109, +0.0044], p=0.4172; the difference is not significant. Under an approximately matched context budget and a standardized FLAN reader, RECOMP-style sentence packing does not improve the frozen multi-hop baseline, whereas Full context actions retain a positive same-source effect. This is an official-compressor implementation under reader and budget adaptation, not a claimed end-to-end reproduction. Matched tokens also do not create identical structural action spaces: sentence compression and pair-complementary five-document actions optimize different objectives. The approximately 47-token Top-1 condition is retained only as a compatibility diagnostic in the appendix.

## 7. Computational Cost and Deployment Boundary

| System | Generator ms | Selector ms | Reader ms | Total ms | P95 total | Cross scores | Reader calls | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen Top-5 | 0.09 | 0.00 | 140.75 | 140.88 | 252.10 | 0 | 1 | 1.98 |
| Full | 70.05 | 0.61 | 142.59 | 213.48 | 330.56 | 10 | 1 | 2.78 |
| Lite | 6.55 | 0.57 | 136.74 | 143.97 | 254.15 | 0 | 1 | 1.98 |
| Baseline-Truncated-660 | 9.63 | 0.00 | 137.73 | 147.43 | 255.39 | 0 | 1 | 1.90 |
| RECOMP Top1 | 22.04 | 0.00 | 122.65 | 144.76 | 240.11 | 0 | 1 | 2.40 |
| RECOMP 660 | 31.66 | 0.00 | 137.89 | 169.64 | 285.33 | 0 | 1 | 2.40 |

These are measured end-to-end **post-retrieval** times, not reader-only proxies. The shared protocol uses one GPU, batch size one, 50 warmup queries, 500 measured queries, CUDA synchronization, and the same query fingerprint. All online components are recomputed; every final context matches its frozen artifact. Model loading is excluded.

Full's mean is 213.48 ms/query versus 140.88 ms/query for Frozen Top-5, an overhead of 72.60 ms and a 1.52x ratio. Lite lowers the mean to 143.97 ms by removing semantic encoders, but its independent non-inferiority failure prevents promotion to the main method. Every system invokes the answer reader exactly once on the final context. Candidate reader outcomes are offline labels and do not add online reader calls.

Offline work includes action-outcome labeling and fold-specific generator and selector training. Historical offline GPU-hour totals were not recorded and are therefore unavailable. The intended deployment is auditable context organization over a bounded post-retrieval pool. Corpus-scale indexing, dynamic web search, and production streaming are outside the evaluation.

## 8. External Transfer and Calibration

### 8.1 Zero-Shot Frozen Transfer

On 1,000 2WikiMultiHopQA development queries [@ho-etal-2020-2wiki], the unchanged Hotpot gate covers 26.0%. Baseline Answer/SP/Joint F1 are 0.4709/0.4545/0.2463; frozen transfer yields 0.4794/0.4539/0.2496. The deltas are +0.0086/-0.0006/+0.0033. Answer has 95% CI [-0.0021, +0.0191], p=0.1116; SP has [-0.0036, +0.0025], p=0.6928; Joint has [-0.0031, +0.0098], p=0.3296. None is significant, support is effectively flat, and selected answer-drop is 6.92%. This is a failed zero-shot safety-transfer diagnostic.

### 8.2 Few-Shot Gate Calibration

We calibrate only the safety gate using nested K in {16, 32, 64, 128} target-train examples under five fixed seeds. Threshold-only, temperature, Platt, and risk-constrained variants are evaluated with frozen generator, action families, reader, prompt, and evaluation set. The best mean answer-drop is 5.10% at K=128 with threshold_only, 16.26% coverage, Answer/SP/Joint F1 0.4755/0.4542/0.2484, ECE 0.3924, and Brier 0.2335. It misses the pre-specified 4% target. Few-shot calibration partially reduces answer-drop risk but does not recover the in-domain safety level. We do not continue tuning K, seed, temperature, or threshold after observing this failure.

## 9. Analysis

**Opportunity before selection.** The action generator determines whether repair is possible at all. Pair complementarity raises the chance that a proposal contains both hops, while bounded construction prevents opportunity from becoming an uncontrolled permutation search. Selection then trades coverage for answer risk. This explains why conditional gains can exceed population gains without implying a broad treatment effect.

**What the Lite failure means.** Pair complementarity, chains, anchors, and selective safety are the most interpretable mechanisms. Yet the untouched holdout shows that lexical pair features alone do not preserve Full quality within the chosen margin. Missing-hop, MPNet, cross-encoder, and document-opportunity components therefore remain in the stronger implementation. Their mixed individual ablations support neither a claim that each is necessary nor a claim that each always helps.

**Compression versus structured action.** Equalizing token budget removes the most obvious information-volume confound, but it does not equalize objectives. Sentence packing chooses text spans; Full selects a small structural intervention while retaining five-document coverage. The comparison bounds interpretation rather than identifying one universally better constructor.

**Transfer as a gate boundary.** 2Wiki retains some positive candidate opportunity, but the Hotpot safety probabilities are misaligned with target-domain harm. Target-train calibration lowers risk only partially. The current evidence therefore separates reusable action construction from unresolved risk calibration under shift.

## 10. Limitations and Ethical Considerations

The population effects are below one and two F1 points, respectively, on the two same-source holdouts. Selected-query means are conditioned on the policy's choices and cannot be extrapolated to arbitrary or generally improvable queries. Most interventions tie the baseline, and 7.75-7.83% lower Answer F1 among selected interventions.

Both confirmatory sets come from HotpotQA validation. The 2Wiki experiment is non-significant and violates the answer-drop target even after few-shot calibration, so cross-dataset safety remains unresolved. Calibration also requires labeled target-train reader outcomes and is not zero-shot behavior.

Full adds measured online latency relative to Frozen Top-5. Lite reduces this overhead but fails the independent quality rule. Historical offline GPU-hour totals are unavailable. The benchmark begins after retrieval and does not include corpus indexing, network transfer, or retriever execution.

The bounded pool normally contains about ten distractor documents. Larger corpus-scale pools and changing indexes are not evaluated. The fixed reader and support predictor can have entity-, language-, or question-type-specific errors. A selector lowers average risk but offers no correctness guarantee. The method rearranges supplied passages and does not synthesize evidence; this helps auditability but cannot recover facts absent from the pool.

## 11. Conclusion

Reader-aware selection is limited first by the contexts its generator makes possible. Full pair-complementary generation creates bounded two-document alternatives, preserves answer anchors, and exposes them to a fully nested reader-safe selector. Two frozen same-source holdouts show modest positive Answer, SP, and Joint F1 changes, with larger descriptive means on the quarter of queries selected for intervention and exact fallback elsewhere. The Lite failure keeps the full semantic recipe in the primary method while narrowing the conceptual contribution. Equal-budget compression, measured latency, and unsuccessful transfer calibration define clear comparison and deployment boundaries. In short, pair-complementary action generation and fully nested reader-safe selection yield modest but reproducible same-source QA gains, with larger descriptive effects on selected interventions and unresolved cost and transfer boundaries.
