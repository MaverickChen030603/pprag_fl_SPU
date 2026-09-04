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
