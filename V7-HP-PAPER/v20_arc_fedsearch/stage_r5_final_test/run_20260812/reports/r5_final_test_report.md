# V20 R5 One-Shot Final-Test Report

**Decision:** `final_test_strongly_confirmed`

Labels were unsealed once at `2026-08-12T12:07:15.147088+00:00` after all 7,200 predictions passed checksum validation.

## Primary Joint F1

- hotpotqa / flan: +0.0370 [+0.0172, +0.0582], p=0.0003999, W/T/L=32/257/11
- hotpotqa / unifiedqa: +0.0280 [+0.0121, +0.0442], p=0.0007998, W/T/L=28/266/6
- 2wikimultihopqa / flan: +0.0420 [+0.0214, +0.0636], p=0.0003999, W/T/L=42/245/13
- 2wikimultihopqa / unifiedqa: +0.0405 [+0.0191, +0.0632], p=0.0003999, W/T/L=39/246/15
- musique / flan: +0.0291 [+0.0124, +0.0469], p=0.0012, W/T/L=27/265/8
- musique / unifiedqa: +0.0225 [+0.0059, +0.0405], p=0.004399, W/T/L=23/267/10

R4 macro delta: +0.0387; R5 macro delta: +0.0332.
Support rescue/harm: 86/2; rescue mean Joint delta: +0.2695.

SP uses the shared frozen V16 support predictor and is context-level, not an independent cross-reader replication.

The first evaluator invocation completed all frozen CSVs but failed while serializing a NumPy integer into the final JSON. This report was recovered exclusively from those frozen CSVs without reopening labels or recomputing metrics.
