## 8. External Transfer and Calibration

### 8.1 Zero-Shot Frozen Transfer

On 1,000 2WikiMultiHopQA development queries [@ho-etal-2020-2wiki], the unchanged Hotpot gate covers 26.0%. Baseline Answer/SP/Joint F1 are 0.4709/0.4545/0.2463; frozen transfer yields 0.4794/0.4539/0.2496. The deltas are +0.0086/-0.0006/+0.0033. Answer has 95% CI [-0.0021, +0.0191], p=0.1116; SP has [-0.0036, +0.0025], p=0.6928; Joint has [-0.0031, +0.0098], p=0.3296. None is significant, support is effectively flat, and selected answer-drop is 6.92%. This is a failed zero-shot safety-transfer diagnostic.

### 8.2 Few-Shot Gate Calibration

We calibrate only the safety gate using nested K in {16, 32, 64, 128} target-train examples under five fixed seeds. Threshold-only, temperature, Platt, and risk-constrained variants are evaluated with frozen generator, action families, reader, prompt, and evaluation set. The best mean answer-drop is 5.10% at K=128 with threshold_only, 16.26% coverage, Answer/SP/Joint F1 0.4755/0.4542/0.2484, ECE 0.3924, and Brier 0.2335. It misses the pre-specified 4% target. Few-shot calibration partially reduces answer-drop risk but does not recover the in-domain safety level. We do not continue tuning K, seed, temperature, or threshold after observing this failure.
