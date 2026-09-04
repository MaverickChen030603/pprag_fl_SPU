# 2WikiMultiHopQA Frozen-Transfer Report

- Protocol: fully frozen HotpotQA V4 generator, selector, thresholds, coverage, reader prompt, and support threshold.
- Dataset: deterministic label-blind sample of 1000 2Wiki dev queries.
- Opportunity: 317/1000 queries (31.70%); positive-action density 14.29%.
- Selector coverage: 26.00%; selected-action answer-drop rate 6.92%.
- Answer F1 delta: +0.0086.
- Supporting-fact F1 delta: -0.0006.
- Joint F1 delta: +0.0033.
- Frozen-transfer success rule: True.

This is a cross-dataset zero-shot transfer result. No 2Wiki label, reader outcome, threshold, or coverage value was used for generation or selection.
