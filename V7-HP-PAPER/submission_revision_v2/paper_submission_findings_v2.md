# Reader-Safe Context Action Selection for Multi-Hop Question Answering

## Abstract

Upstream retrieval improvements do not necessarily yield a better ordered context for a downstream reader. We study this policy-action-to-reader gap as a selective context-intervention problem. Given a frozen top-5 context and a bounded candidate pool, an organizer either preserves the baseline or applies one fixed reordering or one-document insertion. A two-stage selector first estimates answer safety and then scores evidence/product opportunity. All outcome-derived features, model choices, thresholds, action-family restrictions, and intervention coverage are fitted with fully nested five-fold query-level cross-fitting.

On a reproducibly sampled, HotpotQA-derived 1,000-query evaluation with a fixed FLAN-T5-large reader, title-level support recall@5 improves from 0.8190 to 0.8310 (+0.0120; p=0.007) and title-level support F1 from 0.7483 to 0.7633 (+0.0150; p=0.018). Answer F1 changes by +0.0028 (p=0.344) and the custom answer-title-support product by +0.0079 (p=0.1245); neither is significant. Removing the nested safety feature reverses the answer-F1 sign. The larger ceiling is candidate opportunity: 797/1,000 queries have no positive action in the main-eligible table. The result supports reader-safe evidence organization as a control point, but not an official HotpotQA joint, cross-reader, cross-dataset, privacy, or end-to-end federated claim.

## 1. Introduction

Multi-hop QA requires a reader to combine evidence from more than one document [@yang-etal-2018-hotpotqa]. A retrieval pipeline can surface useful documents and still fail at the final interface: top-k truncation can remove one hop, an insertion can displace an answer-bearing anchor, and reordering can place key evidence where the reader uses it poorly [@liu-etal-2024-lost]. Irrelevant context can also degrade reasoning [@shi-etal-2023-distracted]. We call the mismatch between upstream policy activity and the context actually used by the reader the **policy-action-to-reader gap**.

We isolate the last step. The organizer receives a frozen baseline top-5 context and a small action table. It can fall back, reorder the same documents, or insert one candidate with paired displacement. It cannot synthesize text. Every action is therefore executable, attributable, and directly comparable under one fixed reader.

The challenge is to learn from reader outcomes without contaminating held-out queries. Our fully nested protocol fits an answer-safety nuisance model and final selector solely inside each outer training split. This replaces the query-excluded but not fold-pure estimate in v1.

## 2. Related Work

Set-oriented retrieval addresses collective evidence sufficiency rather than independent passage relevance [@lee-etal-2025-shifting]. RankRAG aligns ranking and generation [@yu-etal-2024-rankrag], while Reader-Centered Passage Selection uses reader behavior to identify useful passages [@xin-etal-2025-aligning]. Prompt compressors such as LLMLingua, LongLLMLingua, and RECOMP reduce unhelpful context or selectively augment a reader [@jiang-etal-2023-llmlingua; @jiang-etal-2024-longllmlingua; @xu-etal-2024-recomp]. Our method is less expressive: it selects one bounded structural edit relative to a frozen baseline.

Fallback is related to reject-option learning [@geifman-elyaniv-2019-selectivenet], except that the system always answers and abstains only from editing. Federated search motivates our upstream setting [@dhasade-etal-2026-ragroute; @mao-etal-2025-fede4rag], but the evaluated organizer is centralized over candidate text and uses synthetic client identities.

## 3. Method

For query `q`, the baseline is an ordered five-document context `C0(q)`. Each candidate action produces another five-document context by a fixed reorder or insertion/replacement. The selector uses retrieval-score changes, lexical and bridge proxies, action shape, displacement, anchor preservation, and a predicted answer-safety probability. Target-query gold answers and supporting facts are unavailable at inference.

Training actions are labeled by running the fixed reader. An action is positive when answer F1 does not decrease, the answer-title product increases, and either title recall increases or title F1 does not decrease. The primary selector is constrained rather than scalar-weighted: reject actions below a predicted safety threshold, score positive opportunity among survivors, apply only the train-selected intervention coverage, and otherwise fall back.

For each of five outer folds, 800 queries form outer train and 200 form outer test. Five inner folds produce outer-train OOF safety predictions. A nuisance model fitted to all outer-train queries predicts outer test. Configuration and coverage are chosen on outer train and frozen. All folds choose the same two-stage configuration at 0.5 coverage. Audits show zero split overlap and no outer-test outcome use.

## 4. Experimental Setup

We reconstruct 1,000 examples from Hugging Face `hotpot_qa/distractor`, validation split, by seed-44 shuffle followed by first-1,000 selection. The ordered IDs reproduce exactly. Five actions per query yield 5,000 action rows; the two-document action is excluded from the main selector.

The reader is `google/flan-t5-large`, with a 3,200-character context, tokenizer limit 1,024, 32 generated tokens, one beam, and no sampling. We report answer EM/F1 plus custom title-level support recall/F1 and the per-query answer-title-F1 product. These are not official HotpotQA supporting-fact or joint metrics because predicted sentence IDs were not generated. Significance uses 2,000 paired bootstrap resamples.

## 5. Results

**Table 1: Fully nested 1,000-query result.** Evidence columns are custom title-level metrics.

| Method | Answer access | Title recall | Title F1 | Answer F1 | Answer-title product |
|---|---:|---:|---:|---:|---:|
| Frozen baseline | 0.8330 | 0.8190 | 0.7483 | 0.6100 | 0.5170 |
| Fully nested selector | **0.8410** | **0.8310** | **0.7633** | 0.6127 | 0.5249 |
| Difference | +0.0080 | +0.0120 | +0.0150 | +0.0028 | +0.0079 |

| Metric | 95% CI | p |
|---|---:|---:|
| Answer F1 | [-0.0102,+0.0161] | 0.3440 |
| Title recall | [+0.0025,+0.0215] | 0.0070 |
| Title F1 | [+0.0011,+0.0289] | 0.0180 |
| Answer-title product | [-0.0061,+0.0225] | 0.1245 |

The evidence metrics improve significantly; answer and product do not. The selector changes 500 contexts, with 29 answer drops, and preserves 500.

## 6. Analysis

**Table 2: Core nested ablations.**

| Variant | Delta answer F1 | Delta title F1 | Delta product |
|---|---:|---:|---:|
| Primary | +0.0028 | +0.0150 | +0.0079 |
| No nested safety | -0.0029 | +0.0160 | +0.0062 |
| No support features | -0.0023 | +0.0094 | +0.0056 |
| Weighted utility | -0.0031 | +0.0193 | +0.0089 |

Safety gating is not a guarantee, but it is the only tested variant that keeps the aggregate answer delta positive while preserving evidence gain. Weighted utility prefers evidence more aggressively and harms answer F1.

Opportunity dominates ranking error. Across all five templates, 778 queries lack a positive action. Once the excluded two-document action is removed, the main selector has no positive action for 797 queries. A better scorer cannot overcome that ceiling. A diagnostic coverage sweep suggests useful actions beyond the 50% budget, but it is not used to tune the primary result.

## 7. Limitations

The evaluation uses one reader, custom title metrics, one HotpotQA-derived sample, centralized candidate text, synthetic clients, and a restricted action table. The 2Wiki diagnostic does not show transfer. Exact model/tokenizer revisions were not logged. We claim neither official benchmark improvement nor reader robustness, privacy, communication efficiency, natural federated heterogeneity, or cross-dataset generalization.

## 8. Conclusion

A fully nested reader-side organizer can improve title-level evidence coverage while preserving average answer quality, but this experiment does not establish an answer gain. The next credible step is to expand answer-safe candidate opportunity and add official sentence-level support predictions, rather than continue tuning a scorer over a table in which 79.7% of queries have no eligible positive action.
