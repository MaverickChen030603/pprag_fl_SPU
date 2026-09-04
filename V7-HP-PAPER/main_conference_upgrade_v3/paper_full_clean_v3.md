# Reader-Safe Context Action Selection for Multi-Hop Question Answering

## Abstract

Retrieval-augmented multi-hop question answering can fail even when a retrieval pipeline exposes relevant evidence: the final ordered context may omit a complementary hop or displace text that the reader uses to express the answer. Our frozen v2 study formulated this interface as selective bounded context-action choice. A fully nested selector significantly improved title-level support recall and title F1 on 1,000 HotpotQA-derived questions, but did not establish an answer-F1 or answer-title-product gain. Its larger ceiling was candidate opportunity: 797 queries had no answer-safe positive action among four eligible actions.

We test whether a richer, reader-compatible action generator resolves this ceiling. Without using target gold answers, gold support, or reader outcomes, the generator constructs anchor-preserving tail replacements, bridge-aware insertions, bounded two-document chains, redundancy-aware replacements, and restricted reorder-and-insert actions. It produces 7,882 effective contexts, which are evaluated with the same fixed FLAN-T5-large reader and decoding protocol. Positive-query opportunity rises from 20.3% to 23.4%, but remains below our pre-registered 25% continuation floor and 30% main-conference gate. We therefore stop before nested selector, official support, multi-reader, and scale-up experiments. The result narrows the scientific diagnosis: bounded transformations over the same local document pool increase action diversity, especially through two-document chains, but do not sufficiently change which queries admit reader-safe downstream improvements.

## 1. Introduction

Multi-hop QA requires two interfaces to succeed at once. Retrieval must surface evidence for multiple reasoning steps, and a reader must convert the resulting context into an answer. These interfaces are often optimized separately. Retrieval work rewards evidence access or ranking quality; reader work assumes a fixed context. In practice, a document edit that improves support coverage can remove an answer-bearing phrase, shift a useful passage beyond the reader's effective attention, or create an order that a fixed generator handles poorly.

Our earlier v2 experiment isolated this policy-action-to-reader gap. Given a frozen top-five context and a small action table, a constrained selector first estimated answer safety and then chose an evidence-improving action under fully nested query-level cross-fitting. On 1,000 HotpotQA-derived questions, title recall improved by 0.0120 (p=0.007) and title F1 by 0.0150 (p=0.018). Answer F1 changed by only +0.0028 (p=0.344), and the custom answer-title product by +0.0079 (p=0.1245). These results support evidence organization, not improved end-to-end QA.

The v2 opportunity audit revealed a more fundamental limit. Among 4,000 main-eligible actions, only 379 were positive under an answer-safe product criterion, and only 203 queries had at least one such action. For 797 queries, no selector could succeed because the action set contained no admissible improvement. This motivates the question addressed here: can reader-compatible candidate action generation materially raise positive-action opportunity?

We design a bounded deterministic generator rather than an unrestricted combinatorial search. It preserves high-ranked anchors, targets low-value or redundant tail documents, scores candidate complementarity through query, title, entity, and lexical signals, and permits at most one restricted two-document chain or reorder-and-insert operation. All generation features are available at inference and exclude target gold or reader outcomes. The fixed reader evaluates every resulting context, after which a pre-registered opportunity gate decides whether selector training is scientifically justified.

The result is informative but negative for a main-conference upgrade. The generator evaluates 7,882 effective actions and finds 743 positives, yet they cover only 234 queries (23.4%). The +3.1-point increase over v2 confirms that action design matters, but misses the continuation threshold. We stop rather than select a favorable coverage or proceed to expensive evaluations. This turns a vague modeling failure into a sharper boundary: the next bottleneck is not merely selector capacity or the number of local edit templates; it is the source and semantic reach of candidate contexts.

## 2. Related Work

Retrieval-augmented generation combines non-parametric evidence with a parametric reader, while multi-hop QA emphasizes simultaneous access to linked facts. Passage reranking, evidence compression, and reader-aware retrieval all modify the context supplied to the generator. Our setting is closest to reader-centered context selection, but differs in treating intervention as optional and in evaluating the opportunity supplied by an explicit action generator before training a selector.

Selective prediction and risk-coverage analysis motivate our fallback design: when an intervention is uncertain, preserve the baseline. Nested cross-fitting is essential because reader outcomes define both safety and utility labels. Any threshold, feature derived from held-out outcomes, or test-selected intervention rate would overstate generalization. We therefore separate diagnostic-only gold analyses from inference-safe generation and place the opportunity gate before learned selection.

## 3. Reader-Safe Context Action Setting

For each query, the system observes a frozen baseline context of up to five documents and a source pool of HotpotQA distractor documents. An action returns another ordered context with the same budget. Fallback returns the baseline. The reader and decoding protocol are fixed.

An evaluated action is answer-safe when its answer token F1 does not decrease. It is positive when it is answer-safe, increases the per-query answer-title-F1 product, and either improves title recall or does not reduce title F1. These labels are outcome-derived and used only after candidate generation. They measure whether a useful option exists; they are not inference features.

Our primary opportunity metric is the fraction of queries with at least one positive action. Before running v3, we fixed three decision regions: below 25% makes the main-conference upgrade unlikely and stops the downstream pipeline; at least 30% is meaningful; at least 40% is strong.

## 4. Reader-Compatible Candidate Generation

### 4.1 Inference-safe document signals

The generator uses only the question, candidate text, baseline text and rank, and deterministic lexical/routing metadata. Signals include BM25, query-token coverage, title overlap, capitalization-based entity overlap, candidate-to-baseline bridge connection, novel entity coverage, redundancy, and the margin between the added and displaced document's anchor proxy. The output schema is audited for gold answer, support, reader outcome, and oracle fields.

### 4.2 Bounded action families

Anchor-preserving tail replacement keeps the first one to three baseline documents and replaces the lowest-scoring allowed tail. Bridge-aware insertion ranks documents by complementarity to baseline entities and titles rather than query overlap alone. The bounded two-document chain preserves the first three anchors, inserts one complementary pair, and replaces only the final two positions under a stricter lexical safety prior. Redundancy-aware replacement activates only when the baseline contains a duplicate or highly overlapping tail. Joint reorder-and-insert permits one insertion followed by either an anchor-first or bridge-first deterministic order. Duplicate contexts are removed and fallback is always present.

The resulting space averages 7.882 effective actions per query and never exceeds 12. This is larger than v2 but remains auditable and far from an exponential context search.

## 5. Fully Nested Risk-Controlled Selector

The planned v3 selector extends the frozen v2 protocol: five outer query folds, inner out-of-fold answer-safety and opportunity predictions on each outer-training split, train-only safety thresholds, and train-only coverage selection under an answer-delta and answer-drop budget. Outer-test outcomes are unavailable to model fitting and configuration selection.

This stage was not executed. The pre-registered candidate-opportunity gate fired first. This is a methodological result, not missing reporting: training a selector after the generator missed its continuation floor would answer a different, post-hoc question and invite further tuning over a weak action space.

## 6. Experimental Setup

We use the same reproducibly sampled 1,000 HotpotQA distractor-validation questions and frozen baseline contexts as v2. The main reader is `google/flan-t5-large`, revision `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`. Inputs use the same prompt, a 3,200-character context cap, tokenizer limit 1,024, 32 generated tokens, one beam, and no sampling. PyTorch, Transformers, CUDA, model and tokenizer revisions are recorded in the reader manifest.

The v2 paper reports answer EM/F1, custom title recall/F1, and an answer-title product. These title metrics are not official HotpotQA sentence-support or joint metrics. Official evaluation was planned only after the opportunity gate because it requires a nested sentence-support predictor or a reader that emits sentence IDs.

## 7. Main Results

V3 generates 7,882 effective actions plus 1,000 fallbacks. Of the effective actions, 743 (9.43%) are positive and 7300 (92.62%) are answer-safe. At least one positive action exists for 234/1,000 queries (23.4%); 766 remain without a positive option.

Relative to v2, query coverage increases by +3.1%. The gain is real in the descriptive sense, but insufficient under the fixed decision rule. Because 23.4% is below 25%, we do not report a v3 selected downstream score.

## 8. Opportunity, Risk, and Reader Analysis

The diagnostic taxonomy shows that the complete source pool contains all gold-support titles for the 797 frozen no-positive queries, while the older exposed action pool omits a needed support title for 163. Another 386 queries already have perfect baseline answer F1 and title recall, leaving no improvement under the product definition. Ten queries miss both support titles from the baseline, motivating a two-document action.

Among v3 families, bounded two-document chains yield 229 positive actions across 156 queries. Anchor-preserving replacements yield 225 positives, and bridge-aware insertions yield 137. These sets overlap. Their counts show that bounded expressivity recovers cases missed by a single insertion, but also that many positives are concentrated on the same already-improvable queries.

The 92.6% answer-safe action rate is not evidence that the selector problem is easy. Most safe actions are neutral: they preserve an answer but do not improve evidence-product utility. The central requirement is high-density positive opportunity, not safety alone.

## 9. Limitations

The study uses one fixed reader for the v3 opportunity gate and a 1,000-query subset. It does not produce a new official HotpotQA support/joint result, multi-reader result, large-scale validation, or positive external-dataset result. Those omissions follow the registered stop rule and must not be rewritten as unfinished positive evidence.

The generator relies on sparse lexical and capitalization-based entity signals. It does not learn semantic candidate construction, retrieve outside the fixed distractor pool, synthesize missing bridge queries, or jointly optimize context with reader feedback. The unchanged source pool may be the dominant ceiling. Finally, the candidate audit uses gold and outcomes diagnostically; although these values are excluded from generation, the taxonomy itself is not an inference method.

## 10. Conclusion

Reader-safe context selection is constrained first by the actions it is allowed to choose. A bounded reader-compatible generator increases positive-query opportunity from 20.3% to 23.4%, with the strongest contribution from two-document chains, but does not meet the threshold needed for a main-conference upgrade. The safe conclusion remains that fully nested selection improves title-level evidence coverage without a demonstrated answer-quality gain. Future work should change candidate sourcing and semantic construction, then repeat the same no-leak opportunity gate before investing in a stronger selector.
