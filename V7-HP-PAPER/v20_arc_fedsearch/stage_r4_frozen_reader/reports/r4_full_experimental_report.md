# V20 Stage R4: Frozen Dual-Reader End-to-End Evaluation

**Status:** `probe_route_end_to_end_confirmed`

This report evaluates only pre-materialized R3 contexts. It uses the legacy frozen Top-10/Top-5 reader contract, deterministic decoding, and query-level paired bootstrap (5,000 resamples). The centralized retrieval reference is a reference comparator, not a mathematical upper bound.

## Artifacts

- `statistics/main_reader_results.csv`: all formal dataset-reader-method means.
- `statistics/paired_bootstrap.csv`: R1-R0, R2-R0, R2-R1 paired effects.
- `mechanism/support_transition_analysis.csv`: evidence rescue/preservation/harm transmission.
- `gap_recovery/gap_recovery.csv`: centralized-reference gap recovery where the denominator is positive.
