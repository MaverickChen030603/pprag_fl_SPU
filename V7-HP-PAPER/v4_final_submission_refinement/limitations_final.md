# Final Limitations

1. **Opportunity remains incomplete.** The generator passes three of five pre-specified criteria. Overall positive-query coverage is 29.2%, and new-query efficiency remains below the motivating heuristic study.
2. **The strongest holdout is same-source.** The 3,000 queries are disjoint from development and evaluated with a frozen pipeline, but they come from the same HotpotQA source and are not external generalization evidence.
3. **External transfer is non-significant.** On 2WikiMultiHopQA, answer and joint point estimates are positive, supporting-fact F1 is flat, and all relevant confidence intervals include zero. The higher selected answer-drop rate indicates a calibration boundary.
4. **Support replication is not independent.** UnifiedQA receives the same selected contexts and shares the sentence-support predictor used with FLAN. The result supports answer-reader direction but does not independently replicate support prediction.
5. **Generator component evidence is mixed.** Pair complementarity and two-document chains have clear opportunity contributions. Other semantic features are non-monotonic and remain parts of a frozen recipe rather than independently necessary innovations.
6. **The RECOMP comparison has an unmatched output budget.** It uses official code and checkpoint with the same Top-5 input, but emits one sentence under a standardized reader adaptation. The comparison measures compatibility with this setting, not general superiority.
