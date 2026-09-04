# Budget-Matched RECOMP Fairness Report

The original Top-1 setting is retained only as a compatibility diagnostic. The fixed matched protocol ranks sentences with the author-released RECOMP checkpoint and greedily packs whole sentences to the nearest 660-token reader context. The same Top-5 input, FLAN-T5-Large prompt/decoding, and sentence-support predictor are used for every system.

## Development (1,000)

| Method | Tokens | Sentences | Docs | Answer F1 | SP F1 | Joint F1 | Batched reader latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| recomp_top1 | 47.1 | 1.0 | 1.0 | 0.4437 | 0.3701 | 0.2084 | 0.0272 |
| recomp_budget_660 | 637.8 | 14.7 | 4.9 | 0.6049 | 0.4704 | 0.3082 | 0.0272 |
| baseline_truncated_660 | 637.3 | 15.2 | 4.2 | 0.5875 | 0.4899 | 0.3139 | 0.0271 |
| frozen_top5_baseline | 668.2 | -- | 5.0 | 0.6114 | 0.4920 | 0.3241 | see cost report |
| full_v4 | 660.6 | -- | 5.0 | 0.6247 | 0.4973 | 0.3305 | see cost report |

## Frozen 3,000-Query Holdout

| Method | Tokens | Sentences | Docs | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|---:|---:|---:|
| recomp_top1 | 47.7 | 1.0 | 1.0 | 0.4453 | 0.3719 | 0.2178 |
| recomp_budget_660 | 635.9 | 14.7 | 4.9 | 0.6226 | 0.4837 | 0.3259 |
| baseline_truncated_660 | 635.7 | 15.1 | 4.2 | 0.6038 | 0.4904 | 0.3224 |
| frozen_top5_baseline | 664.5 | -- | 5.0 | 0.6183 | 0.4930 | 0.3292 |
| full_v4 | 656.1 | -- | 5.0 | 0.6271 | 0.4987 | 0.3356 |

## Claim Decision

Both stages are complete. On the frozen holdout, 660-token RECOMP differs from the Top-5 baseline by -0.0033 Joint F1 (p=0.4172); this is not a significant advantage. The original Top-1 result is no longer evidence of general superiority. We frame RECOMP and Full as different context-construction objectives under matched reader and context budgets.
