# Appendix: V7-HP Main-Conference Upgrade v3

## A. Frozen v2 metrics

- n=1,000; five outer folds; 500 selected and 500 fallback.
- Title recall delta +0.0120, p=0.007.
- Title F1 delta +0.0150, p=0.018.
- Answer F1 delta +0.0028, p=0.344.
- Answer-title product delta +0.0079, p=0.1245.
- Main-eligible opportunity: 203/1,000 queries.

## B. v3 action inventory

- Effective actions: 7,882.
- Fallbacks: 1,000.
- Effective actions/query: mean 7.882, max 12.
- Positive actions: 743 (0.0943).
- Answer-safe actions: 7300 (0.9262).
- Positive queries: 234 (0.2340).

## C. Leakage separation

Stage 1 taxonomy is diagnostic-only and may use gold support and reader outcomes. Stage 2 is deterministic and consumes only question text, source and baseline text, lexical/entity overlaps, retrieval-style scores, rank, redundancy, and routing metadata. Its serialized output contains no gold answer, gold support, reader outcome, oracle action, or test-fold calibration field.

## D. Stop rule

The observed opportunity 23.4% is below 25%; therefore selector training, official sentence-support evaluation, multi-reader replay, and scale-up were stopped. The external 2Wiki audit independently reached 24.33% opportunity beyond BM25, also below its 25% reader-validation gate.
