# Strong Baseline Proxy Comparison Report

This report compares the frozen v2.3 selector against proxy baselines inspired by set-level selection, reader-aware reranking, and influence/utility-based context selection. These are **not exact reproductions** of SetR, RankRAG, or Influence-Guided Context Selection.

Best proxy/action-level joint_f1_delta row: **v2.3 answer-neutral positive selector** with +0.0150.

The comparison is useful for paper positioning because it tests whether answer-neutral action selection gives benefits beyond simple relevance, set coverage, or utility heuristics using the same candidate-action table.

Safe claim: we compare against proxy implementations inspired by stronger context-selection families. We should not write that we outperform the original SetR/RankRAG/Influence systems.
