# V5 Final Review-Optimized Submission

This directory is a self-contained final anonymous-submission package built from frozen V4/V5 artifacts without holdout retuning.

## Start Here

- `paper_anonymous_v5_final.md`: anonymous main paper.
- `paper_appendix_v5_final.md`: separate appendix.
- `review_response_final.md`: point-by-point response to five review weaknesses.
- `submission_readiness_report.md`: final decision and required flags.
- `final_cost_report.md`: synchronized 500-query online cost benchmark.
- `references.bib`: complete bibliography for cited keys.

## Reproduction

`run_final_submission.sh` runs evidence assembly, paper generation, and final audits. If the frozen cost benchmark is absent, it first runs `run_frozen_cost_benchmark.sh` with 50 warmup and 500 measured queries per system.

All reported online systems call the final FLAN-T5-large reader once per query. Historical offline GPU-hour totals were not recorded and are therefore unavailable.
