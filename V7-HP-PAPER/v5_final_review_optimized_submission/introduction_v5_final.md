## 1. Introduction

Multi-hop question answering (QA) is often framed as finding relevant documents, but a reader consumes an ordered and budget-limited context rather than an unordered relevance set. To answer correctly, that context must expose complementary hops, retain the passage that gives the answer its lexical form, and place the evidence where the reader can use it. More retrieval is therefore not automatically better: adding an individually relevant document can displace an answer anchor or leave two necessary facts disconnected.

This creates a structural limit for post-retrieval selection. A selector can choose only among the contexts proposed by its action generator. If no proposal contains a reader-compatible repair, improved selection cannot help. We call this mismatch the **candidate-opportunity gap**. It is observable as a low density of actions that improve reader outcomes without damaging answer quality.

Fixed expansion templates do not reliably close this gap. Independent insertions and replacements often select documents that are relevant to the question but redundant with each other. Unrestricted replacement can also remove the passage that supplies the answer wording. In our frozen development analyses, increasing action count alone yielded little additional safe positive-query coverage.

We instead organize context construction around **pair complementarity**. The Full Pair-Complementary Action Generator models whether two documents supply different parts of a multi-hop chain, constructs bounded two-document actions, and preserves high-value baseline anchors. Its implementation combines lexical, MPNet, and cross-encoder signals with missing-hop, document-opportunity, and pair-complementarity models. These modules form the empirically stronger Full recipe; our conceptual claim is narrower than a claim that each feature helps monotonically.

A separate reader-safe selector decides whether any generated action should reach the reader. One head predicts answer safety and another predicts positive utility. Frozen thresholds and a coverage budget allow selective intervention; otherwise the system returns the original Top-5 context exactly. The online reader runs once, after this decision, rather than once per candidate action.

We use a fully nested five-fold protocol. Generator and selector training occurs on outer-training queries, thresholds are set using inner out-of-fold predictions, and each outer-test query is processed only by frozen components. We then evaluate the frozen Full system on two disjoint same-source holdouts: 3,000 original holdout queries and an untouched 3,405-query revision holdout. Full improves Answer, SP, and Joint F1 on both. The absolute population changes are modest, while direct paired accounting shows larger descriptive changes on the 25-26% of queries where the policy intervenes and exactly zero change on fallbacks.

The revision experiments also define what we do not claim. A development-frozen Lite-Lexical-Pair simplification fails the independent non-inferiority test, so it does not replace Full. A budget-matched compression comparison is reported as a difference between context-construction objectives, not as a universal rank ordering. Frozen transfer to 2Wiki is non-significant and its few-shot gate calibration misses the safety target. Finally, Full is 1.52x the measured baseline post-retrieval latency, so the result is a quality-risk trade-off over a bounded candidate pool rather than an unqualified deployment claim.

Our contributions are:

1. We formulate the candidate-opportunity gap for reader-aware multi-hop context construction.
2. We introduce pair-complementary, anchor-preserving action generation with bounded two-document chains.
3. We combine the generator with fully nested reader-safe selective intervention and exact fallback.
4. We provide two frozen same-source confirmations plus conditional-effect, non-inferiority, budget-matched compression, online cost, and transfer-boundary analyses.
