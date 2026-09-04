DRAFT — HOTPOTQA SCALE-UP VALIDATED; EXTERNAL CLAIMS PENDING

# Beyond Selection: Opportunity-Aware Context Action Generation for Multi-Hop Question Answering

## Abstract

Reader-side context selection can only exploit actions exposed by its candidate generator. Frozen v2 and v3 studies show that better selection and nearly twice as many heuristic actions do not resolve this candidate-opportunity gap. We introduce a fully nested semantic action generator that estimates the missing evidence type, retrieves complementary documents with bi-encoder and cross-encoder signals, and constructs bounded anchor-preserving actions. V4 evaluates 7,934 effective outer-test actions. Positive-query coverage is 29.2%, conditional non-ceiling coverage is 47.6%, and positive-action density is 14.71%. The fully nested selector intervenes on 260/1000 queries and improves answer F1 by +0.0133 (95% CI [+0.0024, +0.0249], p=0.0176), title recall by +0.0455, and the answer-title product by +0.0442. Under official sentence-level evaluation, answer F1 changes by +0.0133 (p=0.0176), supporting-fact F1 by +0.0053 (p=0.0372), and joint F1 by +0.0064 (p=0.0752). UnifiedQA confirms the direction with answer F1 +0.0129 and joint F1 +0.0088; its answer-drop rate is 1.5%. On 3,000 disjoint same-source queries, the unchanged selector intervenes on 774/3000 queries. FLAN answer F1, supporting-fact F1, and joint F1 improve by +0.0088 (p=0.0096), +0.0056 (p=0.0004), and +0.0064 (p=0.0004); UnifiedQA yields answer/joint gains of +0.0110/+0.0085. The result supports a strengthened main-conference case on HotpotQA, while full opportunity-gate and external-transfer claims remain withheld.

## 1 Introduction

Multi-hop QA often fails before selection: the action table may contain no context that both restores evidence and preserves the reader's answer anchors. V2 established a risk-controlled selector, while v3 showed that expanding fixed templates raises overall coverage only from 20.3% to 23.4% and leaves positive density near 9.4%. This motivates semantic opportunity creation rather than another selector over the same table.

## 2 Method

The v4 system has three generator components: a missing-hop estimator, a semantic document opportunity model, and a pair-complementarity model. Each component is trained on outer-train outcomes and frozen before outer-test generation. Target queries expose only the question, baseline documents, retrieval signals, non-gold entities, and semantic relations. A bounded constructor creates at most eight actions per query across complementary insertion, anchor-preserving replacement, two-document chaining, redundancy replacement, and two order interventions.

## 3 Experimental Protocol

We retain the frozen 1,000-query HotpotQA development set, FLAN-T5-large reader, prompt, context budget, tokenizer limit, and decoding from v2/v3. We report overall and ceiling-aware opportunity, marginal new-query coverage, answer safety, family diversity, and new-query efficiency. Selector, official sentence metrics, second reader, scale-up, and external transfer are strictly gate-controlled.

## 4 Results

V4 evaluates 7,934 effective outer-test actions. Positive-query coverage is 29.2%, conditional non-ceiling coverage is 47.6%, and positive-action density is 14.71%.

The pre-registered gate result is borderline_continue; 3/5 gates passed.

The fully nested selector intervenes on 260/1000 queries and improves answer F1 by +0.0133 (95% CI [+0.0024, +0.0249], p=0.0176), title recall by +0.0455, and the answer-title product by +0.0442.

Under official sentence-level evaluation, answer F1 changes by +0.0133 (p=0.0176), supporting-fact F1 by +0.0053 (p=0.0372), and joint F1 by +0.0064 (p=0.0752).

UnifiedQA confirms the direction with answer F1 +0.0129 and joint F1 +0.0088; its answer-drop rate is 1.5%.

### Frozen Same-Source Scale-Up

On 3,000 disjoint same-source queries, the unchanged selector intervenes on 774/3000 queries. FLAN answer F1, supporting-fact F1, and joint F1 improve by +0.0088 (p=0.0096), +0.0056 (p=0.0004), and +0.0064 (p=0.0004); UnifiedQA yields answer/joint gains of +0.0110/+0.0085.


| Reader | N | Answer F1 baseline | Answer F1 selected | Delta | SP F1 delta | Joint F1 delta | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | 3000 | 0.6183 | 0.6271 | +0.0088 | +0.0056 | +0.0064 | 0.0004 |
| UnifiedQA-T5-Large | 3000 | 0.5662 | 0.5772 | +0.0110 | +0.0056 | +0.0085 | <0.0002 |


## 5 Analysis

V3's net gain of 31 queries decomposes into 81 newly covered v2-negative queries and 50 v2-covered queries not recovered by v3. This distinction motivates breadth-aware reporting. The disjoint 3,000-query run preserves the answer, support, and joint directions under both readers without retuning, reducing the risk that the 1,000-query result is a development-set accident.

## 6 Limitations

The semantic generator is trained from a 1,000-query development study and operates over the available per-query distractor pool. The 3,000-query run is a same-source scale validation, not an external-domain test; second-dataset claims remain unavailable. Title-level evidence metrics are diagnostic proxies and are never renamed as official supporting-fact metrics.

## 7 Conclusion

V4 shows that semantic, query-conditioned action construction can create reader-compatible opportunities beyond fixed templates and transfer them to small but significant answer, support, and joint gains on a disjoint same-source scale-up. Claims remain bounded by the incomplete external validation and the 3/5 opportunity-gate result.
