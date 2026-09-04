# Multi-Reader Claim Boundary

Allowed if a full additional reader succeeds with positive joint/support deltas:

> The frozen v2.3 selected contexts show consistent joint/support-side improvements across readers, while answer_f1 remains reader-sensitive.

Allowed if only a bounded smoke succeeds:

> A bounded reader replication smoke suggests similar joint/support trends, but full multi-reader validation remains future work.

Allowed if no extra reader succeeds:

> Multi-reader replication was prepared by materializing the frozen final_1000 baseline and selected contexts, but additional reader execution was blocked by local model availability and runtime constraints. We therefore keep multi-reader evaluation as a limitation rather than a strengthened claim.

Forbidden:

- v2.3 universally improves all readers
- answer_f1 significantly improves across readers
- multi-reader robustness verified
