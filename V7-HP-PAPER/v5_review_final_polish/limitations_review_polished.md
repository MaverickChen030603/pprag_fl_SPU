## 10. Limitations and Ethical Considerations

1. **Small population effects and added latency.** Full changes Answer/SP/Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 on the two holdouts, while measured post-retrieval latency rises from 140.88 to 213.48 ms/query (1.52x). The evidence supports a quality-risk-cost trade-off, not a broad efficiency claim.

2. **Risk control is not a per-query guarantee.** Reader-safe is an objective label for answer-preservation-oriented selection, not a per-query guarantee. Among selected interventions, 7.75% and 7.83% reduce Answer F1, and 14.86% and 14.19% reduce Joint F1. The selector reduces average risk but cannot guarantee that an individual action will help or tie.

3. **Both confirmatory sets are same-source.** The 3,000- and 3,405-query holdouts are disjoint from development and from each other, but both come from HotpotQA distractor validation. They establish frozen same-source replication, not domain generalization.

4. **External transfer fails its planned criterion.** On 2Wiki, the frozen deltas are non-significant and few-shot calibration reaches a 5.10% selected answer-drop rate rather than the pre-specified 4% target. Cross-dataset risk calibration remains unresolved and requires labeled target-domain reader outcomes.

5. **The candidate pool is bounded.** The study starts from roughly ten Hotpot distractor documents. Pair construction is quadratic in retained set size before pruning, although the frozen system scores ten pairs per query. Corpus-scale retrieval and changing-index behavior are not evaluated.

6. **Support replication is shared.** UnifiedQA changes the answer reader while reusing the same selected contexts and support predictor. It provides directional answer-reader evidence, not independent SP replication; its Joint result also contains the shared support component.

7. **The Lite simplification fails non-inferiority.** Lite reduces measured latency to 143.97 ms/query but is 0.0063 Joint F1 below Full on the independent holdout, with a 95% interval entirely beyond the 0.002 non-inferiority margin. The semantic Full recipe therefore remains necessary for the reported result.

8. **Historical offline cost is incomplete.** The online benchmark is reproducible on one A100 with batch size one, but historical GPU-hour totals for offline outcome labeling and fold-specific training were not recorded. We do not reconstruct them retrospectively.

The method rearranges supplied passages rather than generating evidence. This improves traceability but cannot recover facts absent from the pool. Errors from the fixed answer readers or support predictor may also vary by entity, language, or question type, so deployment in consequential settings requires direct auditing beyond aggregate benchmarks.
