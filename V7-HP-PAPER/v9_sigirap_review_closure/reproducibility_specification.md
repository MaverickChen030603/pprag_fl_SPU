# Reproducibility Specification

## System boundary

Full acts after a frozen retriever has supplied an ordered approximately ten-document pool and a Top-5 baseline. It emits one five-document sequence or returns the baseline byte-for-byte. It does not update the index, modify source text, or search with reader calls at inference.

## Module-by-module contract

### Missing-hop estimator

- **Input:** question-level baseline semantic statistics, baseline CrossEncoder statistics, pair-similarity and redundancy summaries, bridge/entity summaries, question length and entity count.
- **Output:** probabilities for `missing_bridge`, `missing_answer_resolution`, `redundant_context`, `ordering_problem`, and `no_intervention_needed`.
- **Labels:** the five offline missing-type labels on outer-training queries.
- **Model:** standardized balanced multinomial logistic regression; `C` selected from 0.1/1/10 by three-way query-group inner validation; `max_iter=3000`; seed 20260714.

### Document-opportunity model

- **Feature order:** query-document cosine, CrossEncoder relevance, BM25, query-text overlap, query-title overlap, entity overlap, bridge-entity match, novel information, redundancy, maximum/mean baseline semantic similarity, semantic novelty, anchor proxy, normalized source rank.
- **Normalization:** CrossEncoder score is min-max normalized within query, with constant-score queries assigned 0.5; learned vectors then pass through a training-fold `StandardScaler`.
- **Output:** candidate opportunity probability.
- **Model:** balanced logistic regression with the same nested `C` grid, iterations, and seed.

### Pair-complementarity classifier

- **Input order:** left/right document opportunity, query-semantic sum/minimum, CrossEncoder sum/minimum, document-pair cosine, semantic complementarity, entity-chain overlap, bridge sum, novelty sum, redundancy sum.
- **Output:** pair-complementarity probability.
- **Cap:** candidate pairs are scored in the bounded pool; the frozen constructor retains at most three pair proposals and uses ten pair scores per query.
- **Model:** standardized balanced logistic regression with the nested `C` grid and seed 20260714.

### Action families and pruning

- **Families:** single complementary insertion; anchor-preserving replacement; semantic two-document chain; redundancy replacement; bridge-first reorder; answer-anchor-first reorder.
- **Single candidates:** top three ranked candidate documents.
- **Pairs:** top three pair proposals; pair score = `0.65 * learned_pair + 0.175 * left_doc + 0.175 * right_doc`.
- **Generator score:** `0.55 * opportunity + 0.30 * family_probability - 0.15 * removal_risk`.
- **Action cap:** eight. Dynamic cap is four when `no_intervention >= 0.65`, six when `>= 0.45`, otherwise eight.
- **Preservation:** baseline positions 0 and 1 are protected from removal; replacements begin at position 2.
- **Duplicate removal:** byte-equivalent document-ID sequences are deduplicated; every action must have five unique documents.
- **Ordering:** family-specific order is preserved after construction; no target outcome reorders actions.
- **Fallback:** exact original Top-5 order is always present.

### Preservation and utility heads

- **Shared input order:** generator score; new-vs-v3 indicator; mean added-document opportunity; mean added semantic score; removal risk; five missing-type probabilities; six action-family one-hot indicators.
- **Preservation label:** Answer F1 delta `>= 0` relative to the same-query fallback.
- **Utility label:** preservation label is positive, answer-title-product delta is positive, and title recall improves or title F1 does not decrease.
- **Model:** two independent `StandardScaler + LogisticRegression` pipelines; `C=0.5`, balanced classes, `max_iter=2500`, seed 20260714.

### Nested calibration and coverage

- **Outer split:** five query-disjoint folds over 1,000 development queries.
- **Inner split:** five deterministic MD5 query groups per outer fold.
- **Grid:** preservation threshold 0.4/0.5/0.6/0.7/0.8; utility threshold 0.3/0.4/0.5/0.6/0.7; coverage 0.10/0.15/0.20/0.25/0.30.
- **Feasibility:** aggregate development Answer delta `>= -0.001` and selected Answer-drop `<= 0.05`.
- **Objective:** answer-title-product delta plus 0.25 times title-recall delta.
- **Eligibility:** both head thresholds must pass.
- **Tie breaking:** rank eligible actions by utility probability, then preservation probability; Python stable order resolves exact ties.
- **Budget:** `round(coverage * number_of_queries)` per frozen fold; excluded or ineligible queries fall back exactly.
- **Frozen fold configurations:** `(safe, utility, coverage)` = fold 0 `(0.6,0.3,0.30)`, fold 1 `(0.5,0.3,0.30)`, fold 2 `(0.5,0.3,0.25)`, fold 3 `(0.6,0.3,0.15)`, fold 4 `(0.6,0.3,0.30)`.

## Inference-time information audit

Allowed: question text, candidate titles/text, source rank, Top-5 baseline order, lexical/entity features, cached MPNet and CrossEncoder scores computed from query and documents, learned parameters, and frozen thresholds. Prohibited and absent: gold answer, answer-string presence, supporting facts/titles, official metrics, candidate reader outputs, action outcome labels, oracle actions, and target-query fold outcomes.

## Reader and metrics

The primary answer reader is FLAN-T5-Large with greedy decoding, 32 new tokens maximum, input length 1,024, and serialized context capped at 3,200 characters. One final reader call follows selection. The support model is a balanced logistic sentence classifier trained on Hotpot development contexts with threshold 0.7. Official Answer, SP, and Joint metrics are paired by query; confidence intervals and two-sided p-values use 5,000 bootstrap resamples.
