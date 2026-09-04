# Pair-Complementary Context Actions for Multi-Hop Question Answering

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
