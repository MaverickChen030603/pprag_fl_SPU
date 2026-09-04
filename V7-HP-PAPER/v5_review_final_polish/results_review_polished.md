## 5. Main Results

### 5.1 Two Frozen Same-Source Holdouts

| Split | N | Coverage | Baseline A-F1 | Full A-F1 | A delta (95% CI; p) | SP delta (95% CI; p) | Joint delta (95% CI; p) | Selected A-drop |
|---|---:|---:|---:|---:|---|---|---|---:|
| Original frozen holdout | 3000 | 25.8% | 0.6183 | 0.6271 | +0.0088 ([+0.0023, +0.0152]; 0.0096) | +0.0056 ([+0.0031, +0.0083]; 0.0004) | +0.0064 ([+0.0027, +0.0104]; 0.0004) | 7.75% |
| Untouched revision holdout | 3405 | 25.9% | 0.6129 | 0.6244 | +0.0116 ([+0.0052, +0.0178]; <.0002) | +0.0061 ([+0.0036, +0.0088]; <.0002) | +0.0080 ([+0.0044, +0.0116]; <.0002) | 7.83% |

Full improves all three F1 measures on both holdouts. On the original 3,000-query holdout, the paired deltas are +0.0088 Answer, +0.0056 SP, and +0.0064 Joint F1. The untouched 3,405-query holdout confirms +0.0116, +0.0061, and +0.0080. The latter simultaneously serves as the independent Lite non-inferiority test and a replication of Full. Both sets are disjoint from development, Full was frozen before both runs, and revision outcomes were unread when the Lite architecture was fixed. Because both are HotpotQA same-source samples, we do not pool them for a new significance claim.

### 5.2 Descriptive Effects on Policy-Selected Interventions

Population and conditional views answer different questions. The population rows describe the effect of running the frozen policy on every query. The selected rows describe only the contexts that the policy actually changed. They must therefore be interpreted together.

| Holdout | Metric | Population delta | Coverage | Selected mean | Wins/Losses/Ties | Drop rate | Median [Q25, Q75] |
|---|---|---:|---:|---:|---:|---:|---:|
| Original 3,000 | Answer F1 | +0.0088 | 774/3000 (25.8%) | +0.0340 | 89/60/625 | 7.75% | 0 [0, 0] |
| Original 3,000 | SP F1 | +0.0056 | 774/3000 (25.8%) | +0.0219 | 123/100/551 | 12.92% | 0 [0, 0] |
| Original 3,000 | Joint F1 | +0.0064 | 774/3000 (25.8%) | +0.0250 | 141/115/518 | 14.86% | 0 [0, 0] |
| Revision 3,405 | Answer F1 | +0.0116 | 881/3405 (25.9%) | +0.0447 | 107/69/705 | 7.83% | 0 [0, 0] |
| Revision 3,405 | SP F1 | +0.0061 | 881/3405 (25.9%) | +0.0237 | 127/94/660 | 10.67% | 0 [0, 0] |
| Revision 3,405 | Joint F1 | +0.0080 | 881/3405 (25.9%) | +0.0309 | 169/125/587 | 14.19% | 0 [0, 0] |

The zero medians and interquartile ranges show that most selected contexts tie the baseline. Answer F1 decreases on 60 of 774 original-holdout interventions and 69 of 881 revision-holdout interventions; Joint F1 decreases more often. In both holdouts, every fallback context and metric is exactly identical to the baseline. Although the selected subset has larger mean deltas, most selected contexts tie the baseline and some are harmful; the conditional result characterizes the policy's chosen subset rather than an oracle-improvable population.

### 5.3 Full-to-Lite Non-Inferiority

| Revision-holdout system | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|
| Frozen Top-5 | 0.6129 | 0.4862 | 0.3201 |
| Full | 0.6244 | 0.4923 | 0.3280 |
| Lite-Lexical-Pair | 0.6149 | 0.4860 | 0.3217 |

Lite minus Full Joint F1 is -0.0063 (95% CI [-0.0104, -0.0023], p=0.0004). It misses both the point and interval versions of the frozen 0.002 margin. Lite reduces computation, but the independent quality criterion fails; it is therefore a simplification diagnostic rather than a replacement for Full.
