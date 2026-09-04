# Final Multi-Reader Decision

## Answers

1. Context materialization succeeded: **pass**. `num_missing_context=0`, baseline/selected contexts are available for 1000 examples.
2. Local reader availability: **reader_available**. Usable readers: `['google/flan-t5-large', 't5-base']`.
3. Extra reader replication completed: **False**.
4. Joint/support positivity on extra readers: **not evaluated**.
5. answer_f1 remains reader-sensitive / unverified beyond the frozen main reader.
6. Placement: **limitation / appendix attempt**.
7. Submission target: **Findings / COLING**.

## Runtime Feasibility

- max GPU free MiB: 11340
- runtime decision: cpu_smoke_or_stop

## Final Decision

`limitation_only`

If no extra reader completed, do not continue patching experiments now. The current experiments are sufficient for paper writing as a HotpotQA-centered paper with 2Wiki diagnostic limitation and multi-reader as a limitation/replication-ready appendix attempt.
