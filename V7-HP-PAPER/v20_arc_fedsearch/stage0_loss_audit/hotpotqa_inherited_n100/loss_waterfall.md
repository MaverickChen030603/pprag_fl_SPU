# Stage U0 Loss Waterfall (Inherited Replay)

This report is an offline gold-label decomposition over frozen V17 development pools. 
It does not train or evaluate a router, it does not call a reader, and rank-percentile 
merging is a label-free calibration baseline rather than a learned calibration result.

| Stage | Complete-support rate | Delta from previous |
|---|---:|---:|
| Centralized reference @20 | 0.660 | -- |
| Actual selected clients cover all evidence | 0.760 | +0.100 |
| Actual local retrieval within selected clients | 0.440 | -0.320 |
| Transmission retention @15 | 0.440 | +0.000 |
| Raw merge complete support @10 | 0.440 | +0.000 |
| Rank-percentile merge complete support @10 | 0.410 | -0.030 |
| Reader context complete support @5 (raw) | 0.410 | +0.000 |

- Queries: 100
- Routing absence after centralized reference: -0.100
- Local retrieval absence after routing: +0.320
- Raw merge loss at Top-10: 0.000
- Rank-percentile rescue at Top-10: 0.000

## Scope Limitation
The inherited pool records local depth five.  This replay can diagnose the existing `Bc=3, local-k=5` contract, 
but cannot decide a `local-k>5` document-allocation policy.  V20 must materialize a fresh all-client local-k=10 
pool before Stage U1/U2 conclusions or reader evaluation.
