# Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## Abstract

Context selection cannot improve a multi-hop question when its candidate action set contains no useful, reader-compatible alternative. A controlled heuristic expansion nearly doubles the number of context actions on HotpotQA yet raises positive-query opportunity only from 20.3% to 23.4%, exposing a candidate-opportunity gap. We address this gap with a fully nested pipeline that first generates bounded context actions using missing-hop estimates, semantic document opportunity, pair complementarity, and answer-anchor-preserving construction, and then applies an action only when a reader-safe selector predicts it to be useful without harming the answer. The generator raises positive-query opportunity to 29.2% and positive-action density from 9.43% to 14.71%, although two of five pre-specified opportunity criteria remain unmet. On a 1,000-query development protocol, official answer F1 improves by 0.0133 (p=0.0176) and supporting-fact F1 by 0.0053 (p=0.0372), while joint F1 shows a positive non-significant trend (+0.0064, p=0.0752). Without further tuning, the frozen pipeline improves answer, supporting-fact, and joint F1 by 0.0088, 0.0056, and 0.0064 on 3,000 disjoint same-source HotpotQA queries; the confirmatory joint result is significant (p=0.0004). Answer and joint directions are consistent for FLAN-T5-Large and UnifiedQA-T5-Large. A frozen 2WikiMultiHopQA transfer yields positive but non-significant answer and joint changes and flat support F1, which bounds rather than establishes cross-dataset generalization. These results show that semantic opportunity creation and risk-controlled selection can convert selective context changes into small, reproducible reader gains.

## 1. Introduction

Retrieval-augmented question answering usually treats context construction as a ranking problem: retrieve a collection, order it, and ask a reader to answer from the highest-scoring items [@lewis-etal-2020-rag; @xiong-etal-2021-mdr]. Multi-hop questions expose a harder interface. The context must contain multiple complementary facts, preserve the sentence that resolves the answer, and present evidence in a form the reader can use. Adding a relevant passage may help support recall while displacing an answer-bearing passage. Reordering two correct passages may change generation. A document that is individually relevant may be redundant with the existing context, while a lower-ranked document may supply the missing bridge. Evidence availability is therefore necessary but not sufficient for downstream utility.

This interface creates what we call the **policy-action-to-reader gap**: an upstream evidence change does not automatically become a reader gain. It also creates a more basic **candidate-opportunity gap**. A selector can choose only from actions that its generator exposes. If no candidate both repairs the evidence set and preserves answer readability, a more sophisticated selector cannot improve that query. This distinction matters because selection quality is often evaluated after a candidate pool has already fixed the attainable ceiling.

Our motivating studies isolate this ceiling. An initial fixed action table (V2) exposes a positive action for 20.3% of 1,000 HotpotQA development queries [@yang-etal-2018-hotpotqa]. A broader hand-written generator (V3) nearly doubles the table from 4,000 to 7,882 effective actions, but coverage rises only to 23.4% and positive-action density remains effectively unchanged (9.48% to 9.43%). More templates produce more rows, not enough new reader-compatible opportunities. Set-level analysis further shows that V3 newly covers 81 V2-negative queries while losing 50 V2-positive queries. The scientific problem is thus not simply model capacity or action count; it is query-conditioned construction of useful context alternatives.

We introduce a semantic action generator coupled to a reader-safe selector. The generator estimates which reasoning role is missing, scores candidate documents for semantic opportunity, models whether document pairs form complementary hops, and constructs at most eight extractive actions. Actions may insert a complementary document, replace a redundant tail while retaining an answer anchor, construct a two-document chain, remove redundancy, or change a bounded order. The selector then predicts two quantities: whether an action is answer-safe and whether it provides positive joint utility. It intervenes only within a coverage and answer-drop budget chosen on outer-training queries; otherwise it preserves the baseline.

The entire development evaluation is fully nested by query. For each of five outer folds, generator and selector models are trained on 800 queries and applied to 200 disjoint queries. Inner out-of-fold predictions select thresholds and intervention coverage without observing outer-test outcomes. Target-query answers, gold support, reader outcomes, and oracle action labels are excluded from generation and inference. This protocol allows reader outcomes as supervision on training queries while preventing direct target-query leakage.

V4 generates 7,934 effective actions, including 5,655 contexts absent from V3. Positive-action density reaches 14.71%, overall positive-query coverage reaches 29.2%, and coverage among non-ceiling queries reaches 47.63%. The result passes three of five pre-specified opportunity criteria: conditional coverage, marginal breadth, and density. It misses the 30% overall-coverage target by 0.8 points and does not improve new-query efficiency. We therefore interpret the generator as a substantial improvement, not a complete solution.

Downstream results support this bounded interpretation. On the 1,000-query development protocol, the selector intervenes on 26.0% of queries. Official answer F1 improves by 0.0133 and supporting-fact F1 by 0.0053; both are significant in paired bootstrap tests. Joint F1 increases by 0.0064 but is not significant. More importantly, all generator models, selector settings, reader settings, and support thresholds are then frozen and evaluated on 3,000 disjoint queries from the same HotpotQA source. FLAN-T5-Large gains 0.0088 answer F1, 0.0056 supporting-fact F1, and 0.0064 joint F1, all significant under the ordered confirmatory analysis. UnifiedQA-T5-Large shows the same answer and joint direction. This second reader reuses the support predictor, so it is evidence about answer-reader robustness rather than an independent support-pipeline replication.

We further close two comparison gaps. First, an official RECOMP compressor and author checkpoint [@xu-etal-2024-recomp] are evaluated on the same Top-5 inputs under the frozen V4 reader. V4 obtains 0.3305 joint F1 versus 0.2084 for RECOMP; because the reader is standardized rather than copied from the RECOMP paper, we call this a faithful method reproduction with reader adaptation. Second, the complete Hotpot-trained pipeline is transferred without tuning to 1,000 2WikiMultiHopQA development queries [@ho-etal-2020-2wiki]. Answer and joint F1 move positively, supporting-fact F1 is statistically flat, and none of these changes is significant. This is useful boundary evidence, but it does not establish broad cross-dataset generalization.

Our contributions are fourfold:

1. We identify the candidate-opportunity gap: reader-side selectors cannot improve queries for which their action generator exposes no useful intervention.
2. We propose a fully nested semantic action generator that combines missing-hop estimation, document opportunity, pair complementarity, and bounded anchor-preserving context construction.
3. We combine semantic generation with a risk-controlled reader-safe selector trained and evaluated under fully nested query-level cross-fitting.
4. We show significant official answer/support improvements on a 1,000-query development protocol and reproduce significant answer/support/joint gains on a frozen 3,000-query same-source holdout, with consistent answer/joint directions across two readers.

## 2. Related Work

### 2.1 Multi-Hop Retrieval and Evidence Construction

HotpotQA evaluates answer prediction together with supporting facts, making evidence composition observable rather than implicit [@yang-etal-2018-hotpotqa]. 2WikiMultiHopQA expands compositional reasoning types and evidence annotations [@ho-etal-2020-2wiki], while MuSiQue constructs multi-hop questions from controlled single-hop components [@trivedi-etal-2022-musique]. Multi-hop dense retrieval learns iterative retrieval trajectories [@xiong-etal-2021-mdr], but a high-quality retrieval pool still leaves an unresolved interface: which bounded context should a fixed reader actually receive? Our work starts from a frozen per-query document pool and studies context actions inside that pool rather than proposing another corpus-scale retriever.

### 2.2 Reader-Aware Passage and Context Selection

Reader-aware selection methods move beyond stand-alone relevance by optimizing passages for downstream answering. Reader-Centered Passage Selection explicitly aligns passage choice with reader needs [@xin-etal-2025-rcps]. SetR formulates retrieval augmentation as set selection rather than independent ranking [@lee-etal-2025-setr], and RankRAG unifies ranking and generation in a large language model [@yu-etal-2024-rankrag]. Our setting is narrower and complementary. We define a bounded action space over a fixed Top-5 context and separate opportunity generation from selective application. This separation lets us measure whether failure comes from absent useful candidates or from selection among available candidates.

### 2.3 Context Compression and Harmful Evidence

RECOMP learns extractive or abstractive compressors for retrieval-augmented language models [@xu-etal-2024-recomp]. Context-position studies show that relevant information may be underused when placed in the middle [@liu-etal-2024-lost], and irrelevant context can distract otherwise capable models [@shi-etal-2023-distracted]. These findings motivate bounded order changes and answer-anchor protection. Unlike pure compression, V4 may retain, insert, replace, chain, or reorder documents, and may decline to intervene. Our RECOMP comparison isolates the difference between aggressive sentence compression and reader-safe document-level context construction under one reader.

### 2.4 Selective Prediction and Risk-Controlled Intervention

Selective prediction permits a model to abstain when confidence is insufficient [@geifman-elyaniv-2019-selectivenet]. We adapt this logic to context intervention. The fallback is not an unanswered query; it is the original context. Coverage measures how often the system changes that context, while answer-drop rate measures risk among selected actions. This framing prevents support-oriented gains from being purchased through unconstrained answer degradation.

## 3. Problem Setting

### 3.1 Policy-Action-to-Reader Gap

Let q be a query, C0 its frozen baseline context of at most five documents, and R a fixed reader. An upstream policy may produce an alternative context C, but a retrieval-side improvement does not imply that R's answer improves. The reader is sensitive to answer-string availability, evidence order, redundancy, distractors, and the interaction between multiple passages. We call the resulting mismatch between context-side action and answer-side response the policy-action-to-reader gap.

### 3.2 Candidate-Opportunity Gap

For each query, a generator G exposes a bounded action set A(q). An action is positive in the development analysis when it does not reduce answer F1 and increases the product of answer F1 and title-level evidence quality, with a positive title-recall change or non-decreasing title F1. A query has opportunity when at least one effective action is positive. The opportunity of a selector is upper-bounded by the set of queries covered by G. This is the candidate-opportunity gap: selection cannot recover an action that was never constructed.

### 3.3 Bounded Context Actions

V4 acts on document identities and order; it does not synthesize evidence or alter source text. Every context remains within the five-document reader budget. The six effective families are single complementary insertion, anchor-preserving replacement, semantic two-document chain, redundancy replacement, bridge-first reorder, and answer-anchor-first reorder. A fallback preserves C0. The generator emits at most eight effective actions per query. This bounded design permits exhaustive reader evaluation during development and transparent attribution of changes.

### 3.4 No-Leak Evaluation

Reader outcomes supervise models only on training queries. For an outer-test query, generation may use the question, baseline documents, retrieval signals, semantic features, and non-gold text relations. It may not use the target answer, gold supporting facts, target reader outcome, or oracle action quality. Selector thresholds and coverage are chosen from inner out-of-fold predictions on the outer-training split. The 3,000-query holdout and 2Wiki transfer use one frozen pipeline. Generator and selector audits record fold fingerprints and explicitly verify the absence of forbidden fields.

## 4. Semantic Context Action Generation

### 4.1 Missing-Hop Estimation

The missing-hop estimator predicts a distribution over five diagnostic states: missing bridge, missing answer resolution, redundant context, ordering problem, and no intervention needed. Training targets are derived from action outcomes on outer-training queries. At inference, the estimator observes query and context features but no target labels. The state distribution changes the relative priority of insertion, replacement, chain, and order actions rather than deciding the final intervention.

### 4.2 Semantic Document Opportunity Modeling

Each candidate document is represented by lexical retrieval features, MPNet query-document similarity [@song-etal-2020-mpnet], cross-encoder relevance, similarity to existing context documents, and novelty. A logistic opportunity model estimates whether adding the document was useful on outer-training queries. The model is not the final selector: it proposes documents from which actions can be built. This distinction matters in the ablation, where removing the learned document model increases raw opportunity coverage but reduces answer safety, indicating a breadth-risk trade-off rather than monotonic component value.

### 4.3 Pair Complementarity

Multi-hop questions often require two passages whose value emerges jointly. V4 therefore represents candidate pairs using semantic relation, novelty, cross relevance, and opportunity priors. The pair model estimates whether a two-document combination supplies complementary reasoning roles. Removing pair complementarity lowers positive-action density from 14.71% to 10.27% and coverage from 29.2% to 27.7%, the clearest learned-component loss.

### 4.4 Bounded Action Construction

The constructor converts scores into extractive contexts. It can insert a complementary document while retaining high-value anchors, replace a high-risk redundant tail, introduce a scored pair as a two-document chain, remove redundancy, or reorder existing and added documents. The answer anchor is an inference-time lexical/semantic proxy, not a gold-answer check. The generator preserves the five-document budget and deterministic ordering rules. Removing two-document actions reduces coverage to 25.1% and non-ceiling coverage to 40.92%, showing that candidate pairing changes which queries can be helped rather than merely increasing rows.

### 4.5 Generator No-Leak Protocol

The 1,000 queries are partitioned into five outer folds. Each fold's generator is trained on 800 queries and frozen before producing actions for the remaining 200. The final 7,934 outer-test actions contain no target answer, gold support, reader outcome, oracle label, or post-hoc coverage feature. The audit records model hashes and an output SHA-256. Component ablations retrain learned modules on the same outer-training partitions; family removals reuse the corresponding frozen model. No 3,000-query outcome is used for ablation choice.

## 5. Reader-Safe Selection

### 5.1 Answer-Safety Prediction

The first selector head predicts whether an action will avoid reducing answer F1. Features include generator score, novelty relative to V3, added-document opportunity and semantic scores, removal risk, the missing-hop distribution, and action family. A balanced logistic model is trained on outer-training actions. Safety is evaluated as a separate constraint because an action can improve support while deleting or obscuring the answer expression.

### 5.2 Positive-Opportunity Prediction

The second head predicts the positive-action label defined above. It estimates whether an answer-safe action is likely to improve joint answer-evidence utility. At test time, actions must pass both safety and positivity thresholds. Among eligible actions for a query, the selector ranks by predicted positive probability and then safety probability.

### 5.3 Risk-Controlled Coverage and Fallback

Inner training searches safety thresholds, positive thresholds, and coverage levels from 10% to 30%. A configuration is feasible when mean answer F1 does not fall by more than 0.001 and selected-action answer-drop rate is at most 5% on inner out-of-fold data. The objective combines answer-evidence product and title recall. If no action passes or a query falls outside the selected coverage budget, the system uses the original context. Across outer folds the selected coverages are 15%, 25%, or 30%, yielding 26.0% overall intervention.

### 5.4 Fully Nested Cross-Fitting

For each outer fold, five inner query splits generate out-of-fold training predictions. Thresholds and coverage are selected only from those predictions. Models are then fit on all 800 outer-training queries and applied once to the 200 outer-test queries. Fold-level answer-drop rates vary, including one 10% fold, but the aggregate selected-action rate is 5.0%. We report both aggregate risk and fold variation rather than implying uniform calibration.

## 6. Experimental Setup

### 6.1 Data and Development/Holdout Split

The primary dataset is HotpotQA distractor validation [@yang-etal-2018-hotpotqa]. The development evaluation uses a frozen 1,000-query sample and five query-level outer folds. The confirmatory holdout contains the next 3,000 source queries under the frozen seed-44 ordering, is disjoint from development, and exactly reproduces the development baseline for all overlapping audit checks. Because both samples come from the same source split, the second sample is a frozen same-source confirmatory holdout, not a new dataset or a test-set claim.

External validation uses 1,000 examples from 2WikiMultiHopQA development [@ho-etal-2020-2wiki], selected by a deterministic query-ID hash without labels. Each example includes answer, context documents, and support annotations. Only data fields are adapted. Hotpot-trained generator models, selector models, thresholds, coverage policy, FLAN reader, support predictor, and support threshold remain frozen.

### 6.2 Readers

The primary reader is google/flan-t5-large [@raffel-etal-2020-t5], pinned to the recorded revision and evaluated with the frozen V4 prompt, tokenizer budget, and decoding. A second answer reader, UnifiedQA-T5-Large [@khashabi-etal-2020-unifiedqa], receives the same selected contexts. The sentence-support predictor is trained on the 1,000 HotpotQA development protocol and uses threshold 0.7 for development, holdout, and 2Wiki. It is shared across readers; we therefore claim answer-reader directional replication, not independent support replication.

### 6.3 Official and Diagnostic Metrics

Official HotpotQA metrics are answer EM/F1, supporting-fact EM/F1, and joint EM/F1. Title recall, title F1, and answer-title product are development diagnostics used for opportunity labels and selector construction; they are not renamed as official supporting-fact metrics. Opportunity metrics include positive-action density, overall positive-query coverage, conditional coverage among non-ceiling queries, marginal newly covered queries, answer-safe action rate, and new-query efficiency.

### 6.4 Baselines

The frozen baseline is HybridSoftRetriever with dense-sparse mixing alpha 0.55, uniform document weights, and Top-5 output. The 3,000 and 2Wiki runs preserve that baseline rather than substituting BM25 Top-5 [@robertson-zaragoza-2009-bm25]. V2 and V3 serve as controlled motivating generators. For an external method comparison we use RECOMP's official extractive compressor, repository commit 51d4432, and author HotpotQA checkpoint, taking Top-1 sentence from the same five documents [@xu-etal-2024-recomp]. We replace the paper reader with the frozen V4 FLAN reader for comparability and label the result accordingly.

### 6.5 Statistical Testing

All comparisons are paired by query. We report mean differences, 95% intervals, and two-sided p-values from 5,000 paired bootstrap resamples. The development endpoints are answer F1, supporting-fact F1, and joint F1, interpreted as development evidence because the protocol was constructed there. The confirmatory primary endpoint is FLAN official joint F1 on the frozen 3,000-query holdout; answer and supporting-fact F1 are ordered secondary endpoints. UnifiedQA, external transfer, baseline comparison, and ablations are supporting analyses. We do not treat the collection of secondary p-values as one multiplicity-corrected discovery family.

## 7. Results

### 7.1 Action Opportunity

| Method | Effective actions | Positive-action density | Overall positive-query coverage | Non-ceiling coverage | Newly covered vs predecessor | New-query efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 fixed actions | 4,000 | 9.48% | 20.3% | 32.90% | n/a | n/a |
| V3 heuristic expansion | 7,882 | 9.43% | 23.4% | 38.30% | 81 V2-uncovered queries | 0.0209 |
| V4 semantic generation | 7,934 | 14.71% | 29.2% | 47.63% | 81 V3-uncovered queries | 0.0143 |

V3 adds 3,882 actions relative to V2. V4 exposes 5,655 contexts absent from the V3 table. "Newly covered" is a set difference, not the net coverage change: V3 newly covers 81 V2-negative queries but fails to recover 50 V2-positive queries. V4 passes three of five pre-specified opportunity criteria. Overall coverage (29.2% versus a 30% target) and new-query efficiency do not pass.

V4 changes both density and breadth. It finds 1,167 positive actions among 7,934 effective actions, compared with 743 among 7,882 for V3. It covers 292 queries overall and 291 of 611 non-ceiling queries. The answer-safe rate is 92.66%. Conditional coverage, marginal breadth, and density pass the pre-specified criteria. Overall coverage is 0.8 points below its 30% criterion, and efficiency (0.0143 newly covered queries per new context) is below V3's 0.0209 and its multiplier criterion. Thus semantic generation substantially improves the action table, but 708 queries still have no observed positive action.

### 7.2 1,000-Query Development Results

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.483 | 0.6114 | 0.073 | 0.4920 | 0.052 | 0.3241 |
| V4 semantic generation + reader-safe selection | 0.493 | 0.6247 | 0.074 | 0.4973 | 0.051 | 0.3305 |
| Delta | +0.0100 | +0.0133 | +0.0010 | +0.0053 | -0.0010 | +0.0064 |

Paired bootstrap, 5,000 resamples: answer F1 [+0.0024, +0.0249], p=0.0176; supporting-fact F1 [+0.0003, +0.0106], p=0.0372; joint F1 [-0.0005, +0.0132], p=0.0752. Answer F1 and supporting-fact F1 improve significantly at the unadjusted 0.05 level. Joint F1 is positive but not significant. The selector intervenes on 260/1,000 queries and its selected-action answer-drop rate is 5.0%.

The selector changes 260 contexts and falls back on 740. Answer F1 rises from 0.6114 to 0.6247 and supporting-fact F1 from 0.4920 to 0.4973. Joint F1 rises from 0.3241 to 0.3305. The answer and support intervals exclude zero, while the joint interval includes zero. The result therefore supports answer-safe evidence improvement but not a significant development joint claim. Diagnostic title recall increases by 0.0455 and answer-title product by 0.0442, consistent with the selector's training objective.

### 7.3 3,000-Query Frozen Holdout

| Reader | N | Intervention coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Joint F1 95% CI | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | 3,000 | 0.258 | +0.0088 | +0.0056 | +0.0064 | [+0.0027, +0.0104] | 0.0004 |
| UnifiedQA-T5-Large* | 3,000 | 0.258 | +0.0110 | +0.0056 | +0.0085 | [+0.0049, +0.0122] | <0.0002 |

The 3,000 queries are disjoint from the 1,000-query development sample but come from the same HotpotQA distractor validation source. Generator models, selector thresholds and coverage, reader prompts and decoding, and the support threshold were frozen. FLAN answer F1 p=0.0096; FLAN supporting-fact F1 p=0.0004. *UnifiedQA reuses the frozen selected contexts and the same sentence-support predictor; it is an answer-reader replication, not an independent support-pipeline replication.

The unchanged selector intervenes on 774 queries (25.8%). For FLAN, answer F1 rises from 0.6183 to 0.6271, supporting-fact F1 from 0.4930 to 0.4987, and joint F1 from 0.3292 to 0.3356. The primary joint interval excludes zero. The selected-action answer-drop rate is 2.0%, lower than development's 5.0% without threshold adjustment. These results show that the development gains are not restricted to the original 1,000 sample, while remaining a same-source confirmation.

### 7.4 Multi-Reader Directional Replication

UnifiedQA answer F1 increases from 0.5662 to 0.5772 and joint F1 from 0.3045 to 0.3130 on the holdout. Its answer-drop rate is 1.73%. The shared contexts thus help two answer readers in the same direction. Supporting-fact predictions are identical across reader rows because they come from one frozen support model; this design cannot establish reader-independent support replication.

## 8. Analysis

### 8.1 Generator Component Ablations

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

The ablation rejects a simple story in which every semantic score is independently beneficial. Removing missing-hop estimation changes coverage only from 29.2% to 29.0%. Removing MPNet features slightly increases coverage; removing cross-encoder features increases it to 30.6%. The learned document opportunity model is particularly non-monotonic: replacing it with a constant raises raw coverage to 32.6% and density to 14.91% but lowers answer safety by 0.92 points and generates 829 more contexts unseen in V3. Raw opportunity alone therefore does not determine the best risk-controlled downstream pipeline.

Two structural components are robust. Without pair complementarity, positive actions fall from 1,167 to 815 and density to 10.27%. Without two-document chains, coverage falls by 4.1 points and non-ceiling coverage by 6.71 points. Removing anchor-preserving families increases density because the denominator shrinks but decreases query coverage to 27.4%, illustrating why density and breadth must be reported together. Lexical-only and semantic-only feature variants both cover slightly more queries than the frozen full generator, reinforcing that the full system was not selected post hoc from this table. The paper therefore attributes the strongest component evidence to complementary pairs and bounded two-document construction, while treating the remaining scoring modules as one fixed generation recipe.

### 8.2 Opportunity versus Selector Quality

Opportunity is an upper bound, not downstream performance. V4 exposes a positive action for 292 queries but selects actions for 260 based only on inference-safe predictions, and those sets need not coincide. Some uncovered queries cannot be helped by any evaluated action; some covered queries are declined because the predicted risk is high; and some selected actions fail despite the safety model. This decomposition explains why a 5.8-point opportunity increase over V3 becomes a 0.6-to-1.3-point downstream gain. The generator creates possible improvements; the selector decides which are credible without target outcomes.

### 8.3 Risk, Coverage, and Answer Drops

The selector's aggregate development answer-drop rate exactly reaches the 5% budget, while holdout risk falls to 2.0%. External 2Wiki risk rises to 6.92%, showing that safety calibration transfers less cleanly than opportunity generation. This is the principal external failure mode: the frozen selector intervenes at 26%, but 18 selected actions lower answer F1. Selective fallback contains rather than eliminates risk. Future work should calibrate risk under distribution shift without tuning repeatedly on target outcomes.

### 8.4 Ceiling and New-Query Analysis

Of 1,000 development queries, 389 are ceiling cases under the diagnostic definition that baseline answer F1 and title recall both equal one. V4 covers 47.63% of the remaining 611 queries, compared with 38.30% for V3. The set difference matters: V4 covers 81 queries that V3 did not, but its net gain is 58 because coverage also moves among previously positive cases. New-query efficiency falls because semantic generation explores many more distinct contexts. This trade-off is visible in ablations that improve breadth by generating more novel contexts at lower safety.

### 8.5 Failure Cases

Failures fall into four recurring categories. First, no candidate document in the local pool supplies the missing bridge, so bounded construction cannot succeed. Second, the generator finds support evidence but displaces or delays an answer anchor, creating the policy-action-to-reader gap. Third, multiple individually plausible documents form a redundant rather than complementary pair. Fourth, the selector miscalibrates under distribution shift, as reflected in 2Wiki answer drops. These cases motivate broader retrieval pools, explicit multi-objective generation, and shift-aware selective calibration rather than another unbounded template list.

## 9. External Validation and Generalization Boundary

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.402 | 0.4709 | 0.080 | 0.4545 | 0.049 | 0.2463 |
| Frozen V4 transfer | 0.407 | 0.4794 | 0.078 | 0.4539 | 0.047 | 0.2496 |
| Delta | +0.0050 | +0.0086 | -0.0020 | -0.0006 | -0.0020 | +0.0033 |

Answer F1: [-0.0021, +0.0191], p=0.1116. Supporting-fact F1: [-0.0036, +0.0025], p=0.6928. Joint F1: [-0.0031, +0.0098], p=0.3296. The HotpotQA generator, selector, thresholds, coverage, reader, and support predictor are frozen; only the data adapter changes. The result is directionally positive for answer and joint F1, statistically flat for support F1, and not significant. It is external validation evidence, not proof of broad cross-dataset generalization. Opportunity density is 14.29%; positive-query coverage is 31.7%; selection coverage is 26.0%; selected-action answer-drop rate is 6.92%.

The transferred generator retains high opportunity density (14.29%) and covers 31.7% of 2Wiki queries, suggesting that its action construction is not specific to Hotpot query IDs. The selector changes 260 contexts, exactly 26% coverage, without target-data training or threshold adjustment. Answer F1 increases by 0.0086 and joint F1 by 0.0033; supporting-fact F1 changes by -0.0006. All intervals include zero. The correct conclusion is that frozen transfer is directionally positive and statistically non-degrading for answer/joint point estimates, but evidence is insufficient for a broad generalization claim. The higher 6.92% selected answer-drop rate identifies safety calibration as the main transfer boundary.

The faithful-method RECOMP comparison provides a different test. Under the same FLAN reader, RECOMP's Top-1 extracted sentence reduces answer, support, and joint metrics relative to the full Top-5 baseline, while V4 improves them. This does not show that RECOMP is generally inferior: its paper uses a different end-to-end reader setting and optimizes compression. It shows that aggressive one-sentence compression is poorly matched to this fixed multi-hop reader protocol, whereas bounded document-level actions retain complementary evidence.

## 10. Limitations and Ethical Considerations

First, opportunity passes only three of five pre-specified criteria. Overall coverage remains 29.2%, and new-query efficiency is below the V3 reference. The candidate-opportunity gap is narrowed, not solved.

Second, the strongest confirmation uses 3,000 queries from the same HotpotQA source. It is disjoint and frozen but does not measure a new domain. External 2Wiki validation is complete yet inconclusive: answer and joint estimates are positive, support is flat, and intervals include zero.

Third, the generator's component evidence is mixed. Pair complementarity and two-document actions clearly matter, but removing some semantic scorers improves raw opportunity. Since we do not reselect the main pipeline after seeing ablations, the result is honest but leaves room for a cleaner multi-objective generator.

Fourth, UnifiedQA is not an independent support-pipeline replication. It uses the same selected contexts and support predictions as FLAN. A second reader that jointly predicts answers and support, or a separately trained support model, is required for that claim.

Fifth, generator and support models rely primarily on HotpotQA supervision. The 2Wiki transfer changes only data formatting, but its answer-drop rate shows limited safety calibration across datasets. Dataset-specific nested retraining could test architecture-level generalization, but it would be distinct from zero-shot transfer.

Sixth, RECOMP is the only close external method reproduced. It uses official code and checkpoint, but the reader is standardized to V4 and the support metric is extended. The comparison is controlled rather than a comprehensive benchmark against SetR, Reader-Centered Passage Selection, and RankRAG.

Seventh, reader outcomes are used as training supervision. Fully nested cross-fitting prevents target-query outcome leakage, but the method assumes access to outcome-labeled training queries and repeated reader execution. This cost may be substantial for larger readers.

Eighth, all actions are bounded to an available local document pool and extractive text. Missing evidence outside that pool cannot be created. The method does not address corpus-scale retrieval, factuality of generated evidence, or dynamic knowledge updates.

Ninth, no Federated RAG system, privacy mechanism, secure aggregation, or privacy guarantee is evaluated here. Distributed routing motivated the policy-action-to-reader question, but the evidence in this paper concerns centralized context action generation and reader-side selection.

Ethically, selective context intervention can improve traceability because actions retain source documents and support predictions. However, confidence-based fallback may distribute errors unevenly across question types, and external calibration failures may be hidden by average metrics. Releasing per-query decisions, fold manifests, and answer-drop analyses is therefore important. The system should not be used for high-stakes decisions without independent factual verification.

## 11. Conclusion

Context selection has an opportunity ceiling: it cannot choose a reader-compatible action that its generator never exposes. V2 and V3 reveal this ceiling by showing that a larger hand-written table barely changes positive-query coverage. V4 replaces template accumulation with query-conditioned semantic action construction and combines it with fully nested reader-safe selection. The result raises opportunity density and coverage, improves official development answer and supporting-fact F1, and reproduces significant answer, support, and joint gains on a frozen 3,000-query same-source holdout. A second answer reader shows consistent direction. External 2Wiki transfer is promising but non-significant, and component ablations show that better opportunity is a multi-objective breadth-safety problem rather than a monotonic benefit from every semantic score. The central result is therefore bounded but durable: generating better context opportunities, then intervening selectively, is a practical route across the policy-action-to-reader gap.
