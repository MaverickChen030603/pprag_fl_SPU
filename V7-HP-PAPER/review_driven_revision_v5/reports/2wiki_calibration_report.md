# 2Wiki Few-Shot Safety Calibration Report

## Zero-Shot Frozen Transfer

The unchanged HotpotQA gate selects 26.0% of the fixed 1,000-query 2Wiki sample and has a selected answer-drop rate of 6.92%. This result remains the zero-shot transfer result and is not overwritten by calibration.

## Few-Shot Target Calibration

| K | Method | Coverage | Answer-drop | Answer F1 | SP F1 | Joint F1 | ECE | Brier |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | threshold_only | 0.236 | 0.074 | 0.4814 | 0.4544 | 0.2515 | 0.3924 | 0.2335 |
| 16 | temperature_scaling | 0.259 | 0.066 | 0.4818 | 0.4543 | 0.2513 | 0.3996 | 0.2372 |
| 16 | platt_scaling | 0.259 | 0.077 | 0.4861 | 0.4531 | 0.2527 | 0.0300 | 0.0694 |
| 16 | risk_constrained | 0.236 | 0.074 | 0.4814 | 0.4545 | 0.2515 | 0.0300 | 0.0694 |
| 32 | threshold_only | 0.253 | 0.074 | 0.4808 | 0.4528 | 0.2500 | 0.3924 | 0.2335 |
| 32 | temperature_scaling | 0.259 | 0.066 | 0.4818 | 0.4543 | 0.2513 | 0.3880 | 0.2346 |
| 32 | platt_scaling | 0.259 | 0.077 | 0.4861 | 0.4531 | 0.2527 | 0.0343 | 0.0708 |
| 32 | risk_constrained | 0.253 | 0.074 | 0.4808 | 0.4528 | 0.2500 | 0.0343 | 0.0708 |
| 64 | threshold_only | 0.208 | 0.059 | 0.4780 | 0.4530 | 0.2495 | 0.3924 | 0.2335 |
| 64 | temperature_scaling | 0.259 | 0.066 | 0.4818 | 0.4543 | 0.2513 | 0.3877 | 0.2342 |
| 64 | platt_scaling | 0.259 | 0.077 | 0.4861 | 0.4531 | 0.2527 | 0.0192 | 0.0694 |
| 64 | risk_constrained | 0.253 | 0.074 | 0.4806 | 0.4526 | 0.2505 | 0.0192 | 0.0694 |
| 128 | threshold_only | 0.163 | 0.051 | 0.4755 | 0.4542 | 0.2484 | 0.3924 | 0.2335 |
| 128 | temperature_scaling | 0.259 | 0.066 | 0.4818 | 0.4543 | 0.2513 | 0.3868 | 0.2350 |
| 128 | platt_scaling | 0.259 | 0.077 | 0.4861 | 0.4531 | 0.2527 | 0.0123 | 0.0684 |
| 128 | risk_constrained | 0.163 | 0.051 | 0.4755 | 0.4542 | 0.2484 | 0.0123 | 0.0684 |

## Claim Boundary

No setting simultaneously met the <=4% answer-drop, non-decreasing Answer F1, and positive Joint F1 criteria. Safety transfer remains unresolved.
