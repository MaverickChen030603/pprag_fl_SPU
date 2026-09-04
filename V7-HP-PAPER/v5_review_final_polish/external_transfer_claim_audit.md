# External Transfer Claim Audit

## Zero-shot 2Wiki result

| Metric | Baseline | Frozen transfer | Delta | 95% CI | p-value |
|---|---:|---:|---:|---|---:|
| Answer F1 | 0.4709 | 0.4794 | +0.0086 | [-0.0021, +0.0191] | 0.1116 |
| SP F1 | 0.4545 | 0.4539 | -0.0006 | [-0.0036, +0.0025] | 0.6928 |
| Joint F1 | 0.2463 | 0.2496 | +0.0033 | [-0.0031, +0.0098] | 0.3296 |

Coverage is 26.0% and selected Answer-drop is 6.92%. None of the three metric changes is statistically significant.

## Few-shot calibration

The best frozen grid result uses threshold-only calibration with $K=128$: 16.26% coverage, 5.10% selected Answer-drop, Answer/SP/Joint F1 0.4755/0.4542/0.2484. It misses the pre-specified 4% answer-drop target. The search stops without post-failure retuning.

## Allowed claim

2Wiki is a failed transfer and calibration diagnostic. Positive Answer and Joint point estimates motivate future work but do not establish cross-dataset generalization, robustness under shift, or target-domain safety.
