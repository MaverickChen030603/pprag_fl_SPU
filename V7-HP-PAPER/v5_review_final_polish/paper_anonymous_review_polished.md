# Pair-Complementary Context Construction with Risk-Controlled Selection for Multi-Hop QA

## Abstract

Multi-hop question answering depends not only on retrieving relevant documents but also on constructing an ordered context that exposes complementary evidence without displacing answer-bearing passages. We study the **candidate-opportunity gap**: a selector cannot repair a context when its candidate actions omit the needed evidence combination. Our method scores pair complementarity, constructs bounded two-document chains, preserves baseline anchors, and uses fully nested selective intervention with exact fallback. **Reader-safe** denotes an answer-preservation-oriented, risk-controlled selection objective; it does not provide a per-query harm guarantee. On frozen 3,000- and 3,405-query HotpotQA holdouts, the method improves Answer, supporting-fact (SP), and Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080, respectively. The policy intervenes on 25.8% and 25.9% of queries. On the first holdout, selected-query mean deltas are +0.0340/+0.0219/+0.0250, but most interventions tie the baseline and 7.75% reduce Answer F1; these are conditional descriptive results, not causal effects. Measured post-retrieval latency is 213.5 ms/query versus 140.9 ms/query for frozen Top-5, with one final reader call in both. Frozen 2Wiki transfer is non-significant, and few-shot calibration misses its pre-specified answer-risk target. The results support a bounded quality-risk-cost trade-off, not broad efficiency, safety, or cross-dataset claims.

## 1. Introduction

Multi-hop question answering (QA) is often described as finding relevant documents, yet a reader consumes an ordered and budget-limited context. A usable context must expose complementary hops, retain the passage that gives the answer its lexical form, and place evidence where the reader can use it. Adding a relevant document can therefore help support recovery while simultaneously displacing an answer anchor.

This creates a structural limit for post-retrieval selection. A selector chooses only among contexts proposed by its action generator. If no proposal contains a reader-compatible repair, a better selector cannot help. We call this mismatch the **candidate-opportunity gap**. An initial fixed-action study and a later heuristic expansion showed that increasing the number of isolated insertions and replacements did not reliably increase the density of actions that improve downstream reader outcomes without reducing answer quality.

We address this gap with a Full Pair-Complementary Action Generator. It models whether two documents supply different parts of a multi-hop chain, builds bounded two-document actions, and protects high-value passages from the original Top-5 context. The implementation combines lexical, MPNet, and cross-encoder signals with missing-hop, document-opportunity, and pair-complementarity models. These components define the frozen Full recipe; our claim concerns their joint system, not a monotonic benefit from every feature.

A separate two-head selector estimates answer preservation and positive reader utility. Frozen thresholds and a coverage budget permit an intervention only when both conditions pass; otherwise the system returns the original Top-5 context exactly. The reader runs once on the final context. This is an answer-preservation-oriented, risk-controlled objective, not a certification that every selected action is harmless.

We train and evaluate the system with a fully nested five-fold protocol. Generator and selector models fit outer-training queries, thresholds are derived from inner out-of-fold predictions, and outer-test outcomes are never used for architecture or threshold selection. The frozen system is then evaluated on two disjoint same-source HotpotQA holdouts containing 3,000 and 3,405 queries. Both show modest positive population changes in Answer, SP, and Joint F1. Larger means on the roughly one quarter of queries selected for intervention are reported separately with their wins, losses, ties, and drop rates.

The evidence also establishes practical boundaries. Full costs 1.52 times the measured post-retrieval latency of frozen Top-5. A review-driven Lite simplification fails an independently frozen non-inferiority criterion. A budget-controlled RECOMP comparison does not establish a general method ranking. Frozen transfer to 2Wiki is non-significant, and target-domain calibration does not meet its answer-risk target. The evaluated candidate pool is the bounded HotpotQA distractor set, not a corpus-scale retrieval system.

Our contributions are:

1. We formulate the candidate-opportunity gap for reader-aware multi-hop context construction.
2. We introduce pair-complementary, anchor-preserving action generation with bounded two-document chains.
3. We combine the generator with fully nested, risk-controlled selective intervention and exact fallback.
4. We separate population effects from policy-conditional effects and report quality, risk, latency, comparison, reader, and transfer boundaries under frozen protocols.

## 2. Related Work

**Multi-hop retrieval and QA.** HotpotQA and 2WikiMultiHopQA pair answers with supporting facts, making it possible to distinguish answer generation from evidence access [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. Multi-step retrievers such as MDR acquire evidence across retrieval steps [@xiong-etal-2021-mdr]. Our setting starts later: a frozen retriever has already produced a bounded candidate pool, and the method reorganizes that pool for a fixed reader.

**Reader-aware context construction.** Reader-aware retrieval and reranking account for downstream behavior beyond independent query-document relevance [@yu-etal-2024-rankrag; @xin-etal-2025-rcps; @lee-etal-2025-setr]. Prior analyses show that distractors and evidence position can alter reader output [@shi-etal-2023-distracted; @liu-etal-2024-lost]. We distinguish action opportunity from action selection: a risk-controlled selector remains limited by the combinations exposed by its generator.

**Compression and selective prediction.** RECOMP learns extractive context compression [@xu-etal-2024-recomp]. Because its released Hotpot configuration and our near-full contexts have different budgets and objectives, we use the author-released compressor under a common FLAN reader and include a 660-token condition plus a source-order truncation control. Selective prediction motivates fallback when estimated risk is high [@geifman-elyaniv-2019-selectivenet]. Here fallback means preserving the frozen retrieval baseline, and offline reader outcomes supervise the gate without adding online candidate-reader calls.

## 3. Method

### 3.1 Problem and Opportunity

For a question $q$, a frozen retriever returns a bounded document pool $D_q$ and ordered Top-5 baseline $C_0(q)$. A context action maps $C_0$ to another five-document sequence using only documents in $D_q$ and without editing source text. The generator exposes a finite action set $A(q)$; the selector either chooses one action or returns $C_0$.

During training only, an action receives an answer-preservation label when the frozen reader's Answer F1 is no lower than on $C_0$, and a positive-utility label when it improves the reader-side joint answer-evidence utility. Query opportunity is the existence of at least one action satisfying both labels in $A(q)$. It is an empirical upper bound for any selector restricted to that set, not an inference-time label or guarantee.

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

### 3.3 Risk-Controlled Selector

The selector has two balanced logistic heads. The preservation head estimates whether an action retains baseline answer quality; the utility head estimates whether it yields a positive reader-side change. Inner out-of-fold predictions determine both thresholds and a 10-30% coverage budget. On an outer-test query, an action is eligible only if it passes both heads. The highest-ranked eligible action is applied while the fold-level budget remains; all other queries receive the exact baseline. The reader then evaluates one final context. These estimates control average intervention risk but do not certify individual actions.

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

Population and conditional views answer different questions. The population rows describe the effect of running the frozen policy on every query. The selected rows describe only the contexts that the policy actually changed. They must therefore be interpreted together.

| Holdout | Metric | Population delta | Coverage | Selected mean | Wins/Losses/Ties | Drop rate | Median [Q25, Q75] |
|---|---|---:|---:|---:|---:|---:|---:|
| Original 3,000 | Answer F1 | +0.0088 | 774/3000 (25.8%) | +0.0340 | 89/60/625 | 7.75% | 0 [0, 0] |
| Original 3,000 | SP F1 | +0.0056 | 774/3000 (25.8%) | +0.0219 | 123/100/551 | 12.92% | 0 [0, 0] |
| Original 3,000 | Joint F1 | +0.0064 | 774/3000 (25.8%) | +0.0250 | 141/115/518 | 14.86% | 0 [0, 0] |
| Revision 3,405 | Answer F1 | +0.0116 | 881/3405 (25.9%) | +0.0447 | 107/69/705 | 7.83% | 0 [0, 0] |
| Revision 3,405 | SP F1 | +0.0061 | 881/3405 (25.9%) | +0.0237 | 127/94/660 | 10.67% | 0 [0, 0] |
| Revision 3,405 | Joint F1 | +0.0080 | 881/3405 (25.9%) | +0.0309 | 169/125/587 | 14.19% | 0 [0, 0] |

The zero medians and interquartile ranges show that most selected contexts tie the baseline. Answer F1 decreases on 60 of 774 original-holdout interventions and 69 of 881 revision-holdout interventions; Joint F1 decreases more often. In both holdouts, every fallback context and metric is exactly identical to the baseline. Although the selected subset has larger mean deltas, most selected contexts tie the baseline and some are harmful; the conditional result characterizes the policy's chosen subset rather than an oracle-improvable population.

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

Offline work includes action-outcome labeling and fold-specific generator and selector training. Historical offline GPU-hour totals were not recorded and are therefore unavailable. The intended deployment is auditable context organization over a bounded post-retrieval pool. Corpus-scale indexing, dynamic web search, and continuously updated production indexes are outside the evaluation.

### 7.1 Candidate-Pool Scope and Pair Complexity

The method begins after retrieval from the official HotpotQA distractor pool, which contains approximately ten documents per query. In the 3,000-query holdout, 2,973 queries have at least ten available documents, only one has at least twenty, and none has fifty or one hundred. There is therefore no common fixed subset on which to claim large-pool scaling.

Pair construction over a retained set of size $L$ is quadratic before pruning, with at most $L(L-1)/2$ pairs. The frozen protocol sets $L=10$, which gives 45 possible pairs before pruning; the measured deployment scores ten pairs per query. These constants bound the reported latency but do not demonstrate corpus-scale behavior. Future tests would require a separately frozen protocol for subquadratic candidate pairing, approximate nearest-neighbor pair retrieval, adaptive Top-$L$, and calibration under continuously changing indexes.


## 8. External Transfer and Calibration

### 8.1 Zero-Shot Frozen Transfer

On 1,000 2WikiMultiHopQA development queries [@ho-etal-2020-2wiki], the unchanged Hotpot gate covers 26.0%. Baseline Answer/SP/Joint F1 are 0.4709/0.4545/0.2463; frozen transfer yields 0.4794/0.4539/0.2496. The deltas are +0.0086/-0.0006/+0.0033. Answer has 95% CI [-0.0021, +0.0191], p=0.1116; SP has [-0.0036, +0.0025], p=0.6928; Joint has [-0.0031, +0.0098], p=0.3296. None is significant, support is effectively flat, and selected answer-drop is 6.92%. This is a failed zero-shot safety-transfer diagnostic.

### 8.2 Few-Shot Gate Calibration

We calibrate only the safety gate using nested K in {16, 32, 64, 128} target-train examples under five fixed seeds. Threshold-only, temperature, Platt, and risk-constrained variants are evaluated with frozen generator, action families, reader, prompt, and evaluation set. The best mean answer-drop is 5.10% at K=128 with threshold_only, 16.26% coverage, Answer/SP/Joint F1 0.4755/0.4542/0.2484, ECE 0.3924, and Brier 0.2335. It misses the pre-specified 4% target. Few-shot calibration partially reduces answer-drop risk but does not recover the in-domain safety level. We do not continue tuning K, seed, temperature, or threshold after observing this failure.

## 9. Analysis

**Opportunity before selection.** The action generator determines whether repair is possible at all. Pair complementarity raises the chance that a proposal contains both hops, while bounded construction prevents opportunity from becoming an uncontrolled permutation search. Selection then trades coverage for answer risk. This explains why conditional means can exceed population means without implying a broad treatment effect.

**What the Lite failure means.** Pair complementarity, chains, anchors, and selective risk control are the most interpretable mechanisms. Yet the untouched holdout shows that lexical pair features alone do not preserve Full quality within the chosen margin. Missing-hop, MPNet, cross-encoder, and document-opportunity components therefore remain in the stronger implementation. Their mixed individual ablations support neither a claim that each is necessary nor a claim that each always helps.

**Compression versus structured action.** Equalizing token budget removes the most obvious information-volume confound, but it does not equalize objectives. Sentence packing chooses text spans; Full selects a small structural intervention while retaining five-document coverage. The comparison constrains interpretation rather than identifying one universally better constructor.

**Directional answer-reader check.** We replay the same frozen baseline and selected contexts with FLAN-T5-Large and UnifiedQA-T5-Large. FLAN Answer F1 changes from 0.6183 to 0.6271 (+0.0088), while UnifiedQA changes from 0.5662 to 0.5772 (+0.0110). Their Joint F1 point estimates change from 0.3292 to 0.3356 (+0.0064) and from 0.3045 to 0.3130 (+0.0085). Both rows reuse the same support predictor, whose SP F1 changes from 0.4930 to 0.4987 (+0.0056). The second answer reader therefore supplies directional evidence for answer behavior only. It is not an independent SP replication, and the Joint direction is not independent of the shared support component.

**Transfer as a gate boundary.** 2Wiki retains positive Answer and Joint point estimates, but all three tests are non-significant and the Hotpot risk scores are misaligned with target-domain harm. Target-train calibration lowers selected answer-drop only partially. The evidence therefore separates reusable action construction from unresolved risk calibration under shift.

## 10. Limitations and Ethical Considerations

1. **Small population effects and added latency.** Full changes Answer/SP/Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 on the two holdouts, while measured post-retrieval latency rises from 140.88 to 213.48 ms/query (1.52x). The evidence supports a quality-risk-cost trade-off, not a broad efficiency claim.

2. **Risk control is not a per-query guarantee.** Reader-safe is an objective label for answer-preservation-oriented selection, not a per-query guarantee. Among selected interventions, 7.75% and 7.83% reduce Answer F1, and 14.86% and 14.19% reduce Joint F1. The selector reduces average risk but cannot guarantee that an individual action will help or tie.

3. **Both confirmatory sets are same-source.** The 3,000- and 3,405-query holdouts are disjoint from development and from each other, but both come from HotpotQA distractor validation. They establish frozen same-source replication, not domain generalization.

4. **External transfer fails its planned criterion.** On 2Wiki, the frozen deltas are non-significant and few-shot calibration reaches a 5.10% selected answer-drop rate rather than the pre-specified 4% target. Cross-dataset risk calibration remains unresolved and requires labeled target-domain reader outcomes.

5. **The candidate pool is bounded.** The study starts from roughly ten Hotpot distractor documents. Pair construction is quadratic in retained set size before pruning, although the frozen system scores ten pairs per query. Corpus-scale retrieval and changing-index behavior are not evaluated.

6. **Support replication is shared.** UnifiedQA changes the answer reader while reusing the same selected contexts and support predictor. It provides directional answer-reader evidence, not independent SP replication; its Joint result also contains the shared support component.

7. **The Lite simplification fails non-inferiority.** Lite reduces measured latency to 143.97 ms/query but is 0.0063 Joint F1 below Full on the independent holdout, with a 95% interval entirely beyond the 0.002 non-inferiority margin. The semantic Full recipe therefore remains necessary for the reported result.

8. **Historical offline cost is incomplete.** The online benchmark is reproducible on one A100 with batch size one, but historical GPU-hour totals for offline outcome labeling and fold-specific training were not recorded. We do not reconstruct them retrospectively.

The method rearranges supplied passages rather than generating evidence. This improves traceability but cannot recover facts absent from the pool. Errors from the fixed answer readers or support predictor may also vary by entity, language, or question type, so deployment in consequential settings requires direct auditing beyond aggregate benchmarks.

## 11. Conclusion

Multi-hop context selection is limited by the actions its generator exposes. Pair-complementary construction creates bounded two-document alternatives, preserves baseline answer anchors, and submits an action to a fully nested, risk-controlled selector with exact fallback. Two frozen same-source HotpotQA holdouts show modest positive population changes, while policy-selected queries have larger mean changes but mostly tie the baseline and include measurable harm. Full also costs 1.52 times the measured post-retrieval latency. The failed Lite non-inferiority test, non-significant 2Wiki transfer, bounded candidate pool, and shared support predictor define the present scope. The contribution is therefore a controlled method and evaluation for improving context opportunity under a fixed reader, together with an explicit account of when its quality gains do and do not justify intervention.
