## 3. Problem and Method

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

### 3.4 Training supervision and inference contract

Candidate outcomes are produced offline by replaying the frozen reader on actions generated from outer-training queries. An action is labeled preserved when its Answer F1 is no lower than the corresponding baseline and positive when its joint answer-evidence utility increases. These labels supervise document, pair, preservation, and utility models; they are never features. At inference, the contract is limited to the question, candidate passages, baseline order, lexical/entity/semantic scores, and learned parameters. The system does not inspect gold answers, supporting facts, or candidate reader predictions, and it does not call the reader to search among actions.

This distinction matters for both validity and cost. The policy is reader-aware because reader outcomes define training targets, but deployment is not an expensive per-candidate reader loop. The chosen or fallback context is serialized once and passed to the same frozen answer reader used by baseline.


### 3.5 Component Contract

| Component | Training supervision | Inference inputs | Output |
| --- | --- | --- | --- |
| Missing-hop estimator | Outer-training missing-type labels from offline outcomes | Query and baseline lexical/entity/semantic summaries | Five missing-type probabilities |
| Document opportunity | Outer-training action usefulness labels | Query-document relevance, overlap, novelty, redundancy, bridge and rank features | Candidate opportunity score |
| Pair complementarity | Outer-training pair labels | Two document scores, pair similarity, entity chain, novelty and redundancy | Pair score |
| Action generator | No target-query outcome | Frozen baseline, candidate scores, pair scores, missing-hop state | At most eight unique five-document actions plus fallback |
| Preservation head | Offline Answer F1 non-decrease label | Generator score, removal risk, semantic/opportunity summaries, missing-type and family features | Answer-preservation probability |
| Utility head | Offline answer-compatible utility label | Same inference-safe action features | Positive-utility probability |
| Selector | Inner out-of-fold development predictions | Two probabilities and frozen fold budget | One action or exact baseline fallback |

All learned feature vectors are standardized inside their training pipeline. The supplement gives the exact feature order, labels, regularization, seeds, grids, caps, tie breaking, and fallback rule.

## 4. Fully Nested Protocol

### 4.1 Fully Nested Training and Evaluation

Five outer folds separate training from evaluation. Generator modules and selector heads fit only outer-training queries. Inner folds tune thresholds and coverage without reading outer-test outcomes. Each outer-test query is processed by fold-specific frozen models. The 3,000-query and 3,405-query samples are disjoint from the 1,000 development queries and from each other. They are same-source evaluations, not external replications. No holdout outcome selects Full's architecture, threshold, or coverage rule; the explicit no-leak audit checks query IDs, artifact fingerprints, fold membership, feature availability, and frozen configurations.

### 4.2 Experimental setup

We use HotpotQA distractor validation [@yang-etal-2018-hotpotqa]. A fixed 1,000-query development slice supports nested training and threshold selection. The next disjoint 3,000 queries form the original confirmatory holdout; the remaining 3,405 form an untouched revision holdout. All retain the same source distribution and are not external-domain tests.

The frozen upstream baseline is HybridSoftRetriever with alpha 0.55, uniform document weights, and Top-5 output. FLAN-T5-Large is the primary answer reader [@raffel-etal-2020-t5], using greedy decoding, at most 32 generated tokens, a 1,024-token input limit, and context capped at 3,200 characters. A Hotpot-development support predictor uses a pre-specified frozen 0.7 threshold; a fixed 0.5/0.6/0.7/0.8 post-hoc sensitivity grid is reported without changing the primary threshold. We report official Answer, supporting-fact (SP), and Joint EM/F1. Paired 95% intervals and two-sided p-values use 5,000 query-level bootstrap resamples.

For RECOMP, the author-released HotpotQA compressor scores the same Top-5 input [@xu-etal-2024-recomp]. A 660-token budget is frozen from a development grid; a source-order truncation control uses the same budget. Reader, support predictor, and metrics are shared.

Online cost uses one GPU, batch size one, 50 warmup and 500 measured queries with CUDA synchronization. Model loading and upstream retrieval are excluded; recomputed contexts must match frozen artifacts. Training and candidate labeling are offline.

A fixed ordering and query-ID audit define both frozen samples. The 3,000-query sample is opened after nested development freezing; the remaining 3,405 stay untouched while the Lite architecture and non-inferiority margin are fixed. Query-level paired bootstraps preserve same-question pairing. The second sample does not retune Full.
