# Final Submission Decision

## Status

```yaml
current_v2_status: findings_or_coling_ready
v3_main_conference_status: not_ready
recommended_venue: Findings / COLING-style venue using frozen v2, with v3 as a diagnostic appendix
```

## Why

V3 raises positive-query opportunity from 20.3% to 23.4%, below the pre-registered 25% continuation floor and 30% Gate A. The stop rule was honored, so no v3 downstream selector, official support/joint, multi-reader, or scale-up claim exists.

## Mandatory remaining work

1. Redesign candidate sourcing, not only local context transformations, to exceed 30% opportunity under the same fixed reader and no-leak audit.
2. Re-run the fully nested selector only after that gate passes.
3. Produce sentence-level official HotpotQA support/joint metrics with a nested support predictor or answer-plus-support reader.
4. Freeze selected contexts and verify direction on a second reader.
5. Scale to full validation or at least 3,000 questions without post-scale tuning.

## Optional work

- Learn train-only semantic bridge-query generation.
- Retrieve outside the fixed ten-document distractor pool.
- Revisit 2Wiki only after opportunity exceeds the strong BM25 baseline on at least 25% of the smoke set.
