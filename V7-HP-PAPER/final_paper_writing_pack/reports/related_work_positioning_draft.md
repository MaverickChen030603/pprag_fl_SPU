# Related Work Positioning Draft

## 1. Federated RAG and Federated Search

Prior federated retrieval and RAG studies mainly focus on source routing, client selection, or privacy-preserving aggregation. Our work focuses on what happens after routing: whether a routed context action should be applied to the reader input.

## 2. Multi-hop RAG and Passage Set Selection

Multi-hop RAG requires coherent passage sets rather than isolated relevant passages. Centralized passage selectors optimize relevance and evidence coverage, while our setting must make action decisions under federated routing constraints.

## 3. Reader Sensitivity and Harmful Context

Reader models are sensitive to context composition. Adding support-like evidence can still reduce answer quality if it disrupts answer anchors or introduces distractors.

## 4. Evidence Utility and No-Leak Action Selection

Unlike prior federated RAG studies that mainly focus on source routing, and unlike centralized passage selection methods that optimize passage-set relevance, our work studies the downstream action-selection problem after federated routing. We show that support-like routed contexts can still harm reader answer quality, and propose an answer-neutral positive-action selector that applies only those context actions predicted to preserve answer quality while improving joint/support utility under strict no-leak cross-fitting.
