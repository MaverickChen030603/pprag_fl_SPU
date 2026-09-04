# Extension Summary for Paper

We conducted a high-tier extension audit without modifying the frozen HotpotQA v2.3 result. The extension adds a formal definition of the policy-action-to-reader gap and compares v2.3 against proxy baselines inspired by set-level selection, reader-aware reranking, and influence-style utility selection. These proxies are not exact reproductions of prior systems, but they strengthen the empirical positioning by testing whether the answer-neutral selector improves over relevance/utility-only heuristics under the same candidate-action space.

The multi-reader extension could not be completed because the frozen artifacts contain metrics, titles, and action labels but not the full baseline and selected reader contexts required for safe reader re-evaluation. Therefore the paper should not claim multi-reader robustness in its current form.
