# Reader-Safe Context Action Selection for Multi-Hop Question Answering

## Abstract

Improving retrieval scores does not guarantee a better context for a downstream question-answering reader. Aggregation and top-k truncation can erase upstream routing differences, while a support-oriented edit can remove answer-bearing text or place evidence in an unfavorable order. We study this **policy-action-to-reader gap** at the final context-construction stage. Given a frozen baseline top-5 context and a small candidate pool, our organizer either preserves the baseline or selects one bounded extractive action: a fixed reordering or a one-document insertion with paired displacement. The selector uses only inference-available retrieval, lexical, bridge-proxy, action-shape, and anchor-preservation signals, together with an answer-safety nuisance prediction learned from other training queries.

We evaluate with fully nested five-fold query-level cross-fitting on a reproducibly sampled, HotpotQA-derived set of 1,000 validation queries and a fixed FLAN-T5-large reader. Nuisance safety features, action models, thresholds, action-family restrictions, and intervention coverage are all fitted inside each outer training split. The selector changes 500 contexts and preserves 500. Relative to the frozen baseline, title-level support recall@5 increases from 0.8190 to 0.8310 (+0.0120; paired-bootstrap p=0.007), and title-level support F1 from 0.7483 to 0.7633 (+0.0150; p=0.018). Answer F1 changes from 0.6100 to 0.6127 (+0.0028; p=0.344), and the custom answer-title-support product from 0.5170 to 0.5249 (+0.0079; p=0.1245); neither downstream change is statistically significant. Removing the nested safety feature reverses the answer-F1 sign. The principal bottleneck is opportunity: after excluding a risky two-document template, 797 of 1,000 queries have no answer-safe positive action in the main action set. These findings establish a narrow result: fully nested reader-safe action selection improves title-level evidence coverage without a demonstrated answer-quality gain. They do not constitute an official HotpotQA joint result, a privacy guarantee, or an end-to-end federated-system claim.

## 1. Introduction

Retrieval-augmented generation (RAG) grounds model predictions in externally retrieved documents. In multi-hop question answering, the required evidence is often distributed across multiple documents: one passage identifies a bridge entity and another resolves the answer. HotpotQA was designed around this structure and supplies sentence-level supporting facts for evaluation [@yang-etal-2018-hotpotqa]. Yet most RAG pipelines still expose the reader to a small ordered top-k list. The final answer therefore depends not only on whether relevant documents exist somewhere in a candidate pool, but also on which documents survive truncation, which baseline documents are displaced, and where each piece of evidence appears.

This interface became visible in the V7 experimental line. Upstream experiments changed client upload decisions, hybrid dense-sparse scores, and soft routing weights. On controlled examples, oracle weights could strongly alter contexts and improve reader outcomes. Under strict label-free evaluation, however, policy differences often collapsed into nearly identical top-k inputs. Aggregation, dense representations, and top-k selection flattened upstream distinctions before they reached the reader. We call this mismatch the **policy-action-to-reader gap**.

The gap is not only a retrieval problem. A context with an additional gold support title can still yield a worse answer if an answer-bearing lexical anchor is removed. Reordering the same documents can also matter: long-context models do not use all positions uniformly [@liu-etal-2024-lost], and irrelevant information can distract reasoning [@shi-etal-2023-distracted]. Reader-centered passage selection similarly argues that relevance to a query and usefulness to a particular reader are not identical [@xin-etal-2025-aligning]. These observations motivate a decision layer that asks whether an available context edit is safe enough to apply.

We isolate that decision. The organizer receives a frozen baseline top-5 context, a bounded candidate action table, and non-gold metadata. It can preserve the baseline, reorder the same documents with a fixed template, or insert one candidate while displacing one baseline document. It does not freely summarize, fuse, delete, or synthesize evidence. This small action space is intentional: every action is an explicit ordered document set, can be executed with the same fixed reader, and can be audited for anchor preservation and displacement.

The key methodological challenge is outcome-derived supervision. To learn which edits are answer-safe, we execute candidate actions on training queries and observe reader outcomes. A naive leave-one-query-out feature can still leak information across an outer evaluation fold. We therefore use fully nested query-level cross-fitting. For each outer fold, answer-safety nuisance models are trained only on the 800 outer-train queries; train-side safety features are inner-fold out-of-fold predictions, while the 200 outer-test queries receive predictions from a frozen outer-train model. Selector choice, thresholds, family restrictions, and intervention coverage are also fitted only on outer train.

Our result is deliberately narrower than the v1 narrative. The fully nested selector significantly improves custom title-level evidence metrics. It does not significantly improve answer F1 or the answer-title-support product. The safety feature matters because removing it changes mean answer F1 from a small positive value to a negative one, but this is an ablation result about preservation, not proof of a large answer gain. Candidate opportunity is a larger limitation: among the four main-eligible actions per query, 797 queries have no action satisfying the paper's answer-safe positive criterion.

The evaluated organizer is centralized over candidate text, and document-to-client identities are synthetic. Distributed and federated retrieval motivate the frozen candidate interface, but the experiment does not validate a private or end-to-end federated system. We therefore remove “Federated” from the title and make the system boundary explicit.

Our contributions are:

1. **Problem formulation.** We identify the policy-action-to-reader gap and formalize reader-side organization as selective, bounded context intervention rather than unconstrained generation.
2. **Reader-safe action protocol.** We implement a constrained two-stage selector that first estimates answer safety, then scores positive evidence/product opportunity, and otherwise falls back to the baseline.
3. **Fully nested evaluation.** We rebuild the outcome-derived nuisance feature, selector, thresholds, and coverage within outer training folds, with machine-readable fold and provenance audits.
4. **Empirical diagnosis.** On 1,000 HotpotQA-derived queries, the method significantly improves title-level evidence metrics but not answer/product metrics; action opportunity, not only ranking error, is the dominant bottleneck.

**Figure 1 placeholder: Scope-aware pipeline.** Distributed/federated candidate sources (motivation and frozen upstream) feed a centralized candidate pool, bounded action generator, supervised reader-safe selector, and fixed reader. The final artwork must mark synthetic client IDs, centralized text, train-only supervision, target-query unavailable information, and fallback.

## 2. Related Work

### 2.1 Multi-hop retrieval and set construction

Multi-hop QA requires evidence that is collectively sufficient rather than independently relevant. HotpotQA combines answer supervision with sentence-level supporting facts [@yang-etal-2018-hotpotqa]. Recent set-oriented retrieval work makes the same distinction explicit: SetR selects passage sets intended to satisfy multiple information needs instead of reranking each passage in isolation [@lee-etal-2025-shifting]. RankRAG aligns ranking and answer generation within one instruction-tuned model [@yu-etal-2024-rankrag]. Our method operates later and is smaller in scope. It assumes candidate production is frozen and chooses one auditable edit to a five-document reader context. We cite these systems as related approaches; the in-project “SetR-style” and “RankRAG-style” diagnostics are not exact reproductions.

### 2.2 Harmful context and context compression

Relevant information can become less usable when placed in the middle of a long input [@liu-etal-2024-lost], and irrelevant context can directly reduce reasoning accuracy [@shi-etal-2023-distracted]. LLMLingua and LongLLMLingua compress prompts while accounting for information density and position [@jiang-etal-2023-llmlingua; @jiang-etal-2024-longllmlingua]. RECOMP trains extractive or abstractive compressors for downstream task utility and can decline augmentation when retrieval is unhelpful [@xu-etal-2024-recomp]. Our bounded actions share the goal of controlling context but avoid generating new text. This yields lower expressivity but clearer provenance and counterfactual action labels.

### 2.3 Reader-aware passage utility

Reader-Centered Passage Selection uses reader behavior to prioritize consistent and useful passages [@xin-etal-2025-aligning]. Our organizer is likewise trained from reader outcomes, but its objective is selective intervention relative to an already functional baseline context. The relevant question is not simply “which passages are best?” but “does replacing or reordering this particular baseline set improve evidence without reducing the answer?”

### 2.4 Selective prediction and fallback

Selective prediction studies systems that abstain when confidence or expected risk is unfavorable. SelectiveNet jointly optimizes prediction and reject behavior under target coverage [@geifman-elyaniv-2019-selectivenet]. Our system does not abstain from answering; it abstains from editing. Fallback preserves the original reader input. This distinction makes intervention coverage and selected-action answer-drop rate natural diagnostics, but our post-hoc risk-coverage curve is not a formal calibration guarantee.

### 2.5 Federated retrieval as motivation

Federated search routes queries among distributed sources, while recent Federated RAG work studies collaborative retriever training and privacy mechanisms [@dhasade-etal-2026-ragroute; @mao-etal-2025-fede4rag; @chakraborty-etal-2025-fedrag-survey]. The present organizer does not implement those protocols. Candidate texts are centrally visible and five client IDs are assigned synthetically. We use the upstream setting to motivate why candidate production and reader use can diverge, not to claim federated privacy, communication efficiency, or natural client heterogeneity.

## 3. Problem Setting

For query `q`, let `C0(q) = [d1,...,d5]` be the frozen ordered baseline context and let `A(q)` be a small set of materialized candidate actions. Executing action `a` produces another ordered five-document context `Ca(q)`. Fallback `a0` leaves `C0` unchanged. A fixed reader `G` maps `(q, Ca)` to an answer.

The organizer observes the query, baseline and candidate text, retrieval/policy metadata, lexical bridge proxies, and action structure. At inference for a target query it cannot observe the gold answer, supporting facts, target reader outcome, oracle action, or support labels. It chooses at most one effective action. The goal is to improve reader-compatible evidence while limiting answer degradation relative to the frozen baseline.

This is a supervised selective-action problem. Training-query outcomes are allowed because the organizer is learned from previously evaluated examples. The statistical requirement is that no outcome from an outer-test query influences its nuisance feature, selector, threshold, action family, or coverage.

### 3.1 Evaluated action space

The table contains five logical rows per query:

1. `top4_bg1_balanced`: fixed reordering of baseline documents;
2. `keep_top2_insert1_slot3`: retain the first two baseline anchors and insert one candidate;
3. `keep_top3_insert1_slot4`: retain the first three and insert one candidate;
4. `keep_top3_bridge_insert1`: insert a bridge-oriented candidate after a preserved prefix;
5. `keep_top3_insert2_strict`: insert two candidates under a stricter template.

The last template is materialized for diagnostics but excluded from the main selector because two simultaneous replacements are harder to make answer-safe. Six no-op cases from the slot-4 generator carry a stored slot-5 fallback name; they do not create an additional logical row. Fallback is a decision, not a sixth action row.

## 4. Method

### 4.1 Inference-safe action features

Each action is represented by six groups of features.

**Retrieval and policy change.** Hybrid-score and agent-weight deltas indicate whether the inserted document receives stronger upstream support than the displaced baseline tail.

**Lexical and bridge proxies.** Query-document overlap, title bridge scores, and sparse cues approximate whether the candidate can connect an intermediate entity. They do not use gold support labels.

**Anchor preservation.** Prefix-preservation flags, top-1/top-2 removal indicators, and displacement scores represent the risk of discarding reader-useful baseline text.

**Action shape.** Added/removed counts and action-family indicators expose whether an edit is a reordering or insertion and how aggressively it modifies the context.

**Evidence proxy.** Support-proxy deltas compare a candidate with the baseline tail and displaced document. These are retrieval-side proxies, not gold target-query support.

**Nested answer-safety prediction.** A nuisance classifier estimates whether the reader answer will avoid degradation. For outer-test actions, this prediction comes from a model trained only on outcomes from the 800 outer-train queries.

### 4.2 Training labels

Every materialized action was previously executed with the same fixed reader. We compute per-query changes in answer F1, title-level support recall, title-level support F1, and their answer-title product. An action is answer-safe when its answer-F1 delta is nonnegative. It is paper-positive when answer F1 does not decrease, the product increases, and either title recall increases or title F1 does not decrease under the archived definition.

These labels define supervised opportunity. They are not inference features for the target query and are not official HotpotQA sentence labels.

### 4.3 Constrained two-stage selector

The v1 system used a weighted listwise target that added product, answer, recall, and title F1. This double-counted related terms and lacked an independent derivation for its weights. The submission-v2 primary objective is weight-free:

1. estimate answer safety and remove actions below a train-selected threshold;
2. score surviving actions for answer-safe positive opportunity;
3. choose the highest-scoring eligible action for each query;
4. intervene only on the outer-train-selected fraction of queries;
5. preserve the baseline otherwise.

The fitted models are lightweight standardized linear scorers. The contribution is the protocol and decision structure, not a claim that linear scoring is algorithmically novel.

### 4.4 Fully nested cross-fitting

We deterministically sort query IDs by integer MD5 and form five 800/200 outer splits. Inside each outer train split, five inner query folds produce out-of-fold nuisance predictions. The nuisance model fitted on all 800 outer-train queries predicts the 200 outer-test queries. Selector configurations are compared only on outer-train OOF data. The frozen configuration then emits outer-test decisions.

All five outer folds independently choose the same configuration: two-stage scoring, safety threshold 0.5, positive threshold 0.1, conservative effective families, and 0.5 intervention coverage. The agreement is observed, not imposed. `nested_feature_audit.json` records zero inner or outer overlap and `outer_test_outcome_used=false` for every fold.

**Figure 2 placeholder: Fully nested cross-fitting.** Outer train produces inner-OOF nuisance features, selector training, and train-only configuration selection; untouched outer test receives only frozen nuisance and selector predictions.

## 5. Experimental Setup

### 5.1 Data and provenance

The main sample contains 1,000 examples from Hugging Face `hotpot_qa`, configuration `distractor`, validation split. We normalize all 7,405 valid rows, shuffle with `random.Random(44)`, and take the first 1,000. Replaying this procedure reproduces both the exact ordered ID sequence and ID set. Checksums, IDs, duplicate audit, and source reconstruction are in `data_manifest.json`; exact folds are in `fold_manifest.json`.

The converted examples retain answer, support titles, and document text. Gold supporting sentence labels were recovered from the source for provenance, but predicted support sentences do not exist for the completed run. Therefore we cannot report official supporting-fact or official joint metrics.

### 5.2 Candidate and client boundary

Five action rows are materialized for every query, yielding 5,000 action examples. Candidate documents are sanitized and then assigned `client_{index mod 5}`. This round-robin identity is synthetic. The organizer sees all candidate text centrally. No non-IID partition, privacy mechanism, or communication evaluation is part of the main result.

### 5.3 Reader

The fixed reader is `google/flan-t5-large`, based on the T5 text-to-text family [@raffel-etal-2020-t5]. The prompt asks the model to answer using only the context and return a short answer. Documents are serialized as numbered `title: text` entries. Context is truncated to 3,200 characters and tokenized to at most 1,024 tokens. Decoding uses `max_new_tokens=32`, one beam, and no sampling. The exact historical Hugging Face model and tokenizer revisions were not logged. Only this reader completed the action table.

### 5.4 Metrics

We report answer access@5, title-level support recall@5, title-level support F1, answer EM/F1, and the mean per-query product of answer F1 and title-level support F1. Legacy artifacts call the last two evidence fields `sp_f1` and `joint_f1`; they are custom title-level metrics, not official HotpotQA supporting-fact or joint scores. The title-F1 implementation includes a full-recall adjustment.

### 5.5 Statistics

We concatenate the 1,000 held-out decisions and use 2,000 paired query-level bootstrap resamples with seed 13. Confidence intervals are percentile intervals. All significance statements apply to the custom metrics defined above.

## 6. Results

### 6.1 Fully nested main result

**Table 1: Fully nested 1,000-query result.** All evidence metrics are title-level custom metrics.

| Method | Answer access@5 | Title recall@5 | Title F1 | Answer EM | Answer F1 | Answer-title product |
|---|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 0.8330 | 0.8190 | 0.7483 | 0.4800 | 0.6100 | 0.5170 |
| Fully nested selector | **0.8410** | **0.8310** | **0.7633** | **0.4810** | **0.6127** | **0.5249** |
| Difference | +0.0080 | +0.0120 | +0.0150 | +0.0010 | +0.0028 | +0.0079 |

The strongest result is evidence-side. Title recall rises by 1.20 percentage points and title F1 by 1.50 points. The answer and product changes are positive in the aggregate but small.

| Metric | Mean delta | 95% paired-bootstrap CI | p-value |
|---|---:|---:|---:|
| Answer F1 | +0.0028 | [-0.0102,+0.0161] | 0.3440 |
| Title recall@5 | +0.0120 | [+0.0025,+0.0215] | 0.0070 |
| Title F1 | +0.0150 | [+0.0011,+0.0289] | 0.0180 |
| Answer-title product | +0.0079 | [-0.0061,+0.0225] | 0.1245 |

Both title-level evidence intervals exclude zero. The answer and product intervals do not. Accordingly, the paper claims improved title-level evidence organization, not improved answer quality or official joint performance.

### 6.2 Fold behavior

Title recall improves in four of five folds and title F1 in four of five. Answer F1 is positive in four folds but falls by 0.0201 in fold 0; the product likewise falls by 0.0173 in that fold. The aggregate evidence signal is significant, but the residual reader harm is not uniformly eliminated.

### 6.3 Action behavior

The selector changes 500 contexts and falls back on 500. All selected actions produce an effective context change. It chooses 415 conservative slot-4 insertions, 51 bridge insertions, 22 reorderings, and 12 slot-3 insertions. Twenty-nine selected actions reduce answer F1, a 5.8% selected answer-drop rate. Fifty-nine selected actions are paper-positive.

## 7. Analysis

### 7.1 Core ablations

**Table 2: Core nested ablations.** Deltas are measured against the same frozen baseline.

| Variant | Delta answer F1 | Delta title recall | Delta title F1 | Delta product | Selected answer-drop rate |
|---|---:|---:|---:|---:|---:|
| Fully nested constrained primary | **+0.0028** | +0.0120 | +0.0150 | +0.0079 | **0.058** |
| Without nested safety feature | -0.0029 | +0.0130 | +0.0160 | +0.0062 | 0.062 |
| Without support features | -0.0023 | +0.0075 | +0.0094 | +0.0056 | 0.064 |
| Inherited weighted utility | -0.0031 | **+0.0150** | **+0.0193** | **+0.0089** | 0.062 |

Removing safety slightly increases evidence deltas but reverses the answer-F1 sign. Removing support features weakens the evidence gains and also yields a negative answer delta. The inherited weighted ranker obtains the largest evidence and product means but harms mean answer F1. These comparisons support a specific design conclusion: separating safety from evidence utility is useful when the baseline answer must be protected.

They do not prove calibrated safety. Twenty-nine selected actions still reduce answers, and the primary answer delta is statistically uncertain.

### 7.2 Candidate opportunity

Across all five materialized templates, 448/5,000 actions are paper-positive and 778 queries have no positive action. Those totals include the two-document insertion excluded from the main selector. In the correct main-eligible scope, 379/4,000 actions are positive, only 203 queries have any positive action, and **797 queries have none**. Excluding the two-document template removes 69 positive rows and eliminates the only positive option for 19 queries.

This distinction changes the diagnosis. A perfect selector over the submitted action set cannot improve 79.7% of queries under the paper-positive criterion. Better scoring can recover missed opportunities among the remaining 203 queries; it cannot create an action that was never materialized. Candidate design is therefore the dominant ceiling.

**Figure 3 placeholder: Main-eligible candidate opportunity.** Show 797 queries with no eligible positive action and 203 with at least one, then annotate selected positives and answer drops with a non-overlapping priority taxonomy.

### 7.3 Risk-coverage diagnostic

The primary 0.5 coverage is selected inside each outer train fold. A post-hoc sweep replays frozen models at target coverages from 0.1 to 1.0. At the 0.5 primary point, answer F1 changes by +0.0028 and product by +0.0079. At realized coverage 0.875, the diagnostic deltas reach +0.0103 and +0.0199, while the answer-drop rate remains near 5.3%. This suggests that useful lower-ranked actions may remain.

We do not promote high coverage to the main result because choosing it after observing held-out outcomes would violate the protocol. The sweep is an analysis and a design hypothesis for a pre-specified future rerun, not formal risk calibration.

### 7.4 Utility sensitivity

Across 27 train-only weight combinations for the inherited scalar utility, title-level gains remain positive, but answer deltas range from -0.0073 to approximately +0.0001. This consistency clarifies why the weighted objective is diagnostic only: changing coefficients does not robustly solve answer preservation. A constrained gate better matches the paper's priority.

### 7.5 External diagnostics

A 2WikiMultiHopQA smoke evaluation showed near-zero deployable transfer despite a larger oracle opportunity. We treat it as evidence that the feature/action design is dataset-specific, not as cross-dataset validation. Earlier HP-hyper and routing diagnostics show small upstream score differences and motivate the policy-action-to-reader gap; they are not comparable baselines for the nested selector.

## 8. Limitations and Ethical Considerations

**Custom metrics.** Title-level support metrics are not official HotpotQA sentence-level supporting-fact metrics. Predicted support sentence IDs were never generated, so official support and joint scoring are unavailable.

**No significant answer gain.** The fully nested answer-F1 and product intervals include zero. The paper establishes evidence-side improvement and a safety-feature ablation, not improved end-to-end answer quality.

**Single reader.** Only FLAN-T5-large completed the action table. Reader dependence is unknown, and the exact historical Hub revision was not logged.

**Synthetic distributed setting.** Client IDs are round-robin and candidate text is centralized. The organizer has no privacy, secure aggregation, differential privacy, communication, or natural non-IID result.

**Restricted actions.** The main selector excludes two-document insertion and does not generate deletion, deduplication, summarization, fusion, or attribution. This restriction supports auditability but creates the 797-query opportunity ceiling.

**Supervised outcome cost.** Training labels require running the fixed reader on candidate actions. This is an offline supervised organizer, not an outcome-free deployment method or online reinforcement learner.

**External validity.** The 2Wiki diagnostic does not support transfer. The 1,000 examples come from one reproducible HotpotQA validation sample rather than the full benchmark.

**Reproducibility boundary.** Data, folds, scripts, environment versions, and checksums are packaged, but exact model/tokenizer revisions and historical raw-reader runtime remain missing. A clean archival tag is also needed because the packaging worktree was dirty.

**Evidence manipulation risk.** A context organizer can hide disagreement or over-prioritize one source by reordering evidence. Real deployments should log the baseline and edited context, preserve source provenance, and expose interventions to auditors.

## 9. Conclusion

This paper isolates a simple but consequential interface in multi-hop RAG: a candidate pool can contain better evidence without delivering a better ordered context to the reader. We formulate the final step as selective bounded action selection and evaluate it under fully nested query-level cross-fitting. The selector significantly improves title-level support recall and title-level support F1 on a 1,000-query HotpotQA-derived sample. It does not establish a significant answer-F1 or answer-title-product gain.

The result narrows the next research problem. Answer-safety estimation prevents the mean answer regression seen in ablations, but most queries have no eligible positive action. Progress is more likely to come from co-designing high-coverage, reader-compatible candidate actions than from repeatedly tuning a scorer over the same table. Any future main-track claim should add official sentence-level support prediction, a pre-specified coverage rerun, more than one reader, and, if federation remains in scope, a genuine distributed client and communication evaluation.
