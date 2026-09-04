# Statistical Claim Plan

## Units and estimands

All comparisons are paired by query. Reported confidence intervals and two-sided p-values use 5,000 paired bootstrap resamples with the fixed experiment seed. Effect sizes are arithmetic mean per-query metric differences.

## Development evaluation

The 1,000-query development sample was used for model and protocol construction through fully nested outer/inner cross-fitting. Official answer F1, supporting-fact F1, and joint F1 are development endpoints. They are reported separately rather than combined into an omnibus success claim. At the unadjusted 0.05 level, answer F1 and supporting-fact F1 are significant; joint F1 is positive but not significant. These p-values are development evidence, not a confirmatory family.

## Frozen same-source confirmatory holdout

The primary confirmatory endpoint is official joint F1 for FLAN-T5-Large on the disjoint 3,000-query same-source holdout. It improves by +0.0064, [+0.0027, +0.0104], p=0.0004. Answer F1 and supporting-fact F1 are ordered secondary endpoints and both improve in the same direction (p=0.0096 and p=0.0004). UnifiedQA is a directional replication family; its answer and joint results are not used to redefine the primary endpoint.

## External and ablation analyses

The 2Wiki frozen-transfer tests are external validation analyses. Their non-significant p-values are reported without converting positive point estimates into a generalization claim. Generator ablations are opportunity analyses; no downstream configuration is selected from the 3,000-query holdout. RECOMP comparisons are controlled baseline analyses. We do not use the large number of secondary p-values to claim a family-wise corrected discovery.
