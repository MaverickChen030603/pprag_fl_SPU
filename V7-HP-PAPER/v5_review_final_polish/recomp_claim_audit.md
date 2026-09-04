# RECOMP Claim Audit

## Frozen budget-controlled comparison

| System | Mean tokens | Represented documents | Answer F1 | SP F1 | Joint F1 | E2E ms/query |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Top-5 | 664.5 | 4.986 | 0.6183 | 0.4930 | 0.3292 | 140.88 |
| Baseline-Truncated-660 | 635.7 | 4.236 | 0.6038 | 0.4904 | 0.3224 | 147.43 |
| RECOMP-660 | 635.9 | 4.873 | 0.6226 | 0.4837 | 0.3259 | 169.64 |
| Full | 656.1 | 4.986 | 0.6271 | 0.4987 | 0.3356 | 213.48 |

RECOMP-660 versus Frozen Top-5 changes Joint F1 by -0.0033, with 95% CI [-0.0109, +0.0044] and p=0.4172. The protocol standardizes the FLAN reader, prompt, support predictor, same Top-5 source input, and an approximately 660-token context budget. Baseline-Truncated controls for source-order packing at the same budget.

## Allowed claim

Under this approximately matched context budget and standardized reader, RECOMP-style sentence packing does not improve the frozen multi-hop baseline. Full has a positive same-source effect under its own structured action objective.

## Prohibited inference

This comparison does not establish that Full generally outperforms RECOMP, that RECOMP is inferior, or that equal tokens create equal structural action spaces. It is an official-compressor implementation under reader and budget adaptation, not an end-to-end reproduction of every RECOMP setting.
