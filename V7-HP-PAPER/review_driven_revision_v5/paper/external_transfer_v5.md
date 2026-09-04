# Pair-Complementary Context Actions for Multi-Hop Question Answering

## External Transfer

### Zero-Shot Frozen Transfer

The unchanged Hotpot gate selects 26.0% of the fixed 1,000-query 2Wiki evaluation sample. Selected answer-drop is 6.92%; Answer/SP/Joint F1 are 0.4794/0.4539/0.2496, with deltas +0.0086/-0.0006/+0.0033. This non-significant, support-flat result is a failed zero-shot safety transfer diagnostic.

### Few-Shot Safety Calibration

Calibration uses only 2Wiki train examples and leaves the generator, reader, prompt, support predictor, positive-opportunity head, and action families frozen.

| K | Method | Coverage | Answer-drop | Answer F1 delta | Joint F1 delta |
|---:|---|---:|---:|---:|---:|
| 16 | threshold_only | 0.236 | 0.074 | +0.0105 | +0.0052 |
| 16 | temperature_scaling | 0.259 | 0.066 | +0.0109 | +0.0050 |
| 16 | platt_scaling | 0.259 | 0.077 | +0.0152 | +0.0064 |
| 16 | risk_constrained | 0.236 | 0.074 | +0.0105 | +0.0052 |
| 32 | threshold_only | 0.253 | 0.074 | +0.0099 | +0.0037 |
| 32 | temperature_scaling | 0.259 | 0.066 | +0.0109 | +0.0050 |
| 32 | platt_scaling | 0.259 | 0.077 | +0.0152 | +0.0064 |
| 32 | risk_constrained | 0.253 | 0.074 | +0.0099 | +0.0037 |
| 64 | threshold_only | 0.208 | 0.059 | +0.0071 | +0.0031 |
| 64 | temperature_scaling | 0.259 | 0.066 | +0.0109 | +0.0050 |
| 64 | platt_scaling | 0.259 | 0.077 | +0.0152 | +0.0064 |
| 64 | risk_constrained | 0.253 | 0.074 | +0.0098 | +0.0041 |
| 128 | threshold_only | 0.163 | 0.051 | +0.0047 | +0.0021 |
| 128 | temperature_scaling | 0.259 | 0.066 | +0.0109 | +0.0050 |
| 128 | platt_scaling | 0.259 | 0.077 | +0.0152 | +0.0064 |
| 128 | risk_constrained | 0.163 | 0.051 | +0.0047 | +0.0021 |

No setting reaches the pre-specified <=4% answer-drop target while preserving Answer and positive Joint F1. Few-shot target calibration therefore does not resolve transfer safety, and no calibrated result is presented as zero-shot generalization.
