# Final Supplement

## A. Frozen Protocol

- Hotpot source ordering seed: 44.
- Development: 1,000 queries.
- Original confirmatory holdout: 3,000 disjoint queries.
- Untouched revision holdout: 3,405 remaining disjoint queries.
- Baseline: HybridSoftRetriever, alpha 0.55, uniform weights, Top-5.
- Reader: FLAN-T5-Large; 3,200 context characters; 1,024 tokenizer positions; greedy 32-token output.
- Support predictor threshold: 0.7.
- Paired bootstrap samples: 5,000.
- Lite Joint-F1 margin: 0.002, frozen before revision outcomes.

## B. Full Component and Training Details

Full combines lexical and entity features, MPNet similarities, cross-encoder relevance, a missing-hop estimator, a document-opportunity model, pair complementarity, anchor-preserving single and two-document actions, and two selector heads. Each learned component is fold-specific. Outcome labels are computed offline from the frozen reader; inference sees no answer, support label, candidate outcome, or oracle action score.

The central concepts are pair complementarity, bounded two-document chains, anchor preservation, and selective risk control. Other modules are included because the joint Full implementation is empirically stronger than Lite, not because every module has been independently validated as a monotonic contribution.

## C. Selected-Policy Distribution

| Holdout | Selected / N | Metric | Mean | Median | Q25 | Q75 | Wins | Losses | Ties | Drop rate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 774/3000 | Answer | +0.0340 | +0.0000 | +0.0000 | +0.0000 | 89 | 60 | 625 | 7.75% |
| Original | 774/3000 | Joint | +0.0250 | +0.0000 | +0.0000 | +0.0000 | 141 | 115 | 518 | 14.86% |
| Revision | 881/3405 | Answer | +0.0447 | +0.0000 | +0.0000 | +0.0000 | 107 | 69 | 705 | 7.83% |
| Revision | 881/3405 | Joint | +0.0309 | +0.0000 | +0.0000 | +0.0000 | 169 | 125 | 587 | 14.19% |

Gain per 100 original-holdout interventions is +3.40 Answer-F1 points and +2.50 Joint-F1 points. These are descriptive accounting quantities, not policy values for unselected queries.

## D. RECOMP Development Curve and Top-1 Diagnostic

The full development curve is reproduced in Section K. The official compressor is evaluated at 64, 128, 256, 384, 512, and 660 token targets on development. The 660-token protocol is frozen for the holdout. The Top-1 compatibility condition averages approximately 47 tokens and one represented document; it is not used for a broad superiority claim. Baseline-Truncated packs source-order sentences to the same targets.

## E. 2Wiki Calibration Grid

The full K-by-method table is reproduced in Section L. Each cell averages five fixed seeds and reports coverage, selected answer-drop, Answer/SP/Joint F1, ECE, and Brier score. The minimum mean answer-drop is 5.10%, above the 4% target.

## F. End-to-End Timing Protocol

Section N reports the component table. The timing harness recomputes every online stage while enforcing frozen final contexts. Full is split into document preprocessing, lexical features, MPNet encoding, cross-encoder scoring, missing-hop prediction, document-opportunity scoring, pair-feature construction, pair-complementarity scoring, action construction, safety, positive utility, serialization, and reader. Full mean/P95 total latency is 213.48/330.56 ms, compared with 140.88/252.10 ms for Frozen Top-5. Online reader calls per query equal one for every system.

## G. Reproducibility Boundary

Frozen method predictions, holdout outcomes, and source artifacts are not overwritten. Historical offline GPU-hour totals were not recorded and are therefore unavailable. The timing result covers post-retrieval execution on one fixed machine and should not be read as a retriever or index benchmark.

## H. Multi-Reader Supporting Analysis

| Answer reader | Baseline Answer F1 | Selected Answer F1 | Answer delta | Baseline SP F1 | Selected SP F1 | SP delta | Baseline Joint F1 | Selected Joint F1 | Joint delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAN-T5-Large | 0.6183 | 0.6271 | +0.0088 | 0.4930* | 0.4987* | +0.0056* | 0.3292 | 0.3356 | +0.0064 |
| UnifiedQA-T5-Large | 0.5662 | 0.5772 | +0.0110 | 0.4930* | 0.4987* | +0.0056* | 0.3045 | 0.3130 | +0.0085 |

The same frozen contexts are replayed for both answer readers. Asterisks mark values from one shared support predictor. The positive Answer F1 direction is therefore the only independently reader-varying observation. The SP values are repeated rather than replicated, and Joint F1 combines each answer reader with that shared support component. We consequently describe this as a directional answer-reader check, not two independent end-to-end reader pipelines.

## I. Candidate-Pool Scope

| Available documents in the 3,000-query holdout | Queries meeting threshold |
|---:|---:|
| At least 10 | 2,973 |
| At least 20 | 1 |
| At least 50 | 0 |
| At least 100 | 0 |

The official distractor pool is approximately ten documents per query. With retained size $L=10$, exhaustive pair formation would create 45 pairs before pruning; the frozen implementation scores ten pairs per query. The reported benchmark is therefore a bounded post-retrieval context-construction test. It does not measure corpus-scale candidate generation, adaptive large-$L$ behavior, or continuously updated indexes.

Potential extensions include subquadratic pair proposals, approximate nearest-neighbor retrieval over pair representations, adaptive Top-$L$ allocation, and risk calibration under changing candidate distributions. Each would require a new frozen protocol rather than extrapolation from the current timing result.


## J. Fully Nested Generator Component Ablations

| Variant | Effective actions | New actions vs V3 | Positive density | Overall coverage | Non-ceiling coverage | New V3-uncovered queries | Training-label preservation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full V4 generator | 7,934 | 5,655 | 14.71% | 29.2% | 47.63% | 81 | 92.66% |
| - missing-hop estimator | 7,952 | 5,619 | 14.47% | 29.0% | 47.30% | 81 | 92.81% |
| - MPNet features | 7,948 | 5,622 | 14.41% | 29.5% | 48.12% | 83 | 92.59% |
| - cross-encoder features | 7,940 | 5,691 | 14.72% | 30.6% | 49.92% | 91 | 92.57% |
| - learned document opportunity model | 7,934 | 6,484 | 14.91% | 32.6% | 53.19% | 110 | 91.74% |
| - pair complementarity | 7,934 | 5,461 | 10.27% | 27.7% | 45.17% | 71 | 93.07% |
| - two-document chain actions | 5,547 | 3,563 | 10.40% | 25.1% | 40.92% | 54 | 93.69% |
| - anchor-preserving families | 5,909 | 4,088 | 16.57% | 27.4% | 44.68% | 73 | 92.45% |
| - redundancy actions | 7,397 | 5,298 | 14.83% | 29.2% | 47.63% | 81 | 92.85% |
| Lexical-only features | 7,952 | 5,652 | 13.87% | 30.7% | 50.25% | 89 | 92.59% |
| Semantic-only features | 7,952 | 5,929 | 14.68% | 30.6% | 49.92% | 97 | 92.48% |

Learned ablations are retrained on each outer-training split and applied to the disjoint outer test fold; structural family removals reuse the frozen fold model. No outcome from the 3,000-query holdout is used. Pair complementarity and two-document actions make the clearest positive contributions. Removing the learned document opportunity model increases raw opportunity coverage to 32.6% but lowers answer safety to 91.74%; lexical-only and semantic-only variants also show that the full generator is not a post-hoc optimum for every opportunity metric. These results support the bounded semantic action space while limiting claims that every scoring submodule is independently necessary. Selector-level V2 diagnostics are reported separately in the appendix because they use a different action table and coverage and therefore are not V4 component ablations.

## K. Full RECOMP Development Budget Curve

This curve is development-only. The 660-token condition was frozen before the 3,000-query holdout.

| Method | Tokens | Sentences | Docs | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|---:|---:|---:|
| RECOMP Top-1 diagnostic | 47.1 | 1.0 | 1.0 | 0.4437 | 0.3701 | 0.2084 |
| RECOMP-64 | 63.1 | 1.4 | 1.3 | 0.4701 | 0.4008 | 0.2302 |
| Baseline-Truncated-64 | 65.2 | 1.5 | 1.0 | 0.3524 | 0.3649 | 0.1670 |
| RECOMP-128 | 127.8 | 2.9 | 2.3 | 0.5255 | 0.4608 | 0.2867 |
| Baseline-Truncated-128 | 127.8 | 3.1 | 1.4 | 0.4508 | 0.4241 | 0.2336 |
| RECOMP-256 | 255.2 | 5.7 | 3.7 | 0.5670 | 0.4772 | 0.3027 |
| Baseline-Truncated-256 | 255.1 | 6.1 | 2.2 | 0.5060 | 0.4648 | 0.2702 |
| RECOMP-384 | 382.9 | 8.6 | 4.4 | 0.5935 | 0.4803 | 0.3152 |
| Baseline-Truncated-384 | 383.8 | 9.1 | 2.9 | 0.5506 | 0.4808 | 0.2989 |
| RECOMP-512 | 507.5 | 11.5 | 4.7 | 0.6043 | 0.4753 | 0.3149 |
| Baseline-Truncated-512 | 508.0 | 12.1 | 3.6 | 0.5735 | 0.4864 | 0.3078 |
| RECOMP-660 | 637.8 | 14.7 | 4.9 | 0.6049 | 0.4704 | 0.3082 |
| Baseline-Truncated-660 | 637.3 | 15.2 | 4.2 | 0.5875 | 0.4899 | 0.3139 |

The Top-1 row is retained only as a compatibility diagnostic. It is not evidence for or against a general compression advantage.

## L. Full 2Wiki Few-Shot Calibration Grid

Each row averages five fixed seeds. No row satisfies the combined pre-specified criterion of at most 4% selected Answer-drop, non-decreasing Answer F1, and positive Joint F1.

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

## M. Fold Thresholds and Outer-Test Behavior

| Fold | Preservation threshold | Utility threshold | Coverage budget | Outer coverage | Outer Answer-drop | Outer Answer delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6 | 0.3 | 0.30 | 0.30 | 0.0167 | +0.0200 |
| 1 | 0.5 | 0.3 | 0.30 | 0.30 | 0.0667 | +0.0022 |
| 2 | 0.5 | 0.3 | 0.25 | 0.25 | 0.1000 | -0.0029 |
| 3 | 0.6 | 0.3 | 0.15 | 0.15 | 0.0333 | +0.0132 |
| 4 | 0.6 | 0.3 | 0.30 | 0.30 | 0.0333 | +0.0339 |

Thresholds come from outer-training/inner predictions; the outer-test values shown here are evaluation results and were not fed back into tuning.

## N. Detailed Latency Components

| System | Component | Mean ms | Median ms | P95 ms |
|---|---|---:|---:|---:|
| Frozen Top-5 | final reader | 140.748 | 124.105 | 251.966 |
| Full | MPNet encoding | 52.867 | 50.072 | 85.833 |
| Full | cross-encoder scoring | 11.235 | 10.734 | 16.398 |
| Full | lexical features | 2.600 | 2.598 | 3.506 |
| Full | pair-feature construction | 1.773 | 1.721 | 2.584 |
| Full | missing-hop prediction | 0.484 | 0.478 | 0.536 |
| Full | action construction | 0.395 | 0.391 | 0.441 |
| Full | preservation head | 0.320 | 0.321 | 0.336 |
| Full | document-opportunity scoring | 0.301 | 0.301 | 0.319 |
| Full | utility head | 0.293 | 0.295 | 0.307 |
| Full | pair-complementarity scoring | 0.292 | 0.290 | 0.313 |
| Full | final reader | 142.593 | 125.286 | 251.423 |
| Full | end-to-end post-retrieval | 213.484 | 199.805 | 330.563 |
| Lite | lexical and pair features | 3.701 | 3.685 | 5.043 |
| Lite | pair scoring/action construction | 2.762 | 2.646 | 3.793 |
| Lite | final reader | 136.738 | 122.384 | 247.132 |
| Lite | end-to-end post-retrieval | 143.967 | 129.687 | 254.147 |
| RECOMP-660 | compressor scoring | 21.936 | 20.769 | 35.487 |
| RECOMP-660 | sentence packing | 9.637 | 9.806 | 12.984 |
| RECOMP-660 | final reader | 137.892 | 120.068 | 246.431 |
| RECOMP-660 | end-to-end post-retrieval | 169.640 | 152.908 | 285.331 |

## O. Venue-Package Interpretation Rules

The 3,000 selected-policy means (+0.0340/+0.0219/+0.0250) and 3,405 selected-policy means (+0.0447/+0.0237/+0.0309) are descriptive conditional summaries. They are not causal effects, oracle opportunity, or expected gains for arbitrary queries. RECOMP-660 is a budget-controlled context-construction comparison only. UnifiedQA changes the answer reader but shares contexts and the support predictor. All candidate-outcome labels are offline and absent at inference.

## P. No-Leak Manifest Checklist

- Development IDs, original holdout IDs, and second-holdout IDs are disjoint.
- Generator and selector training uses outer-training queries only.
- Thresholds and coverage are set from inner out-of-fold predictions.
- No holdout outcome changes architecture, thresholds, or action families.
- Test-time features exclude answers, supporting facts, and candidate reader outcomes.
- Fallback contexts are byte-for-byte equivalent to the frozen Top-5 representation used by the reader.
- All reported online systems make one final answer-reader call per query.

## Q. Outcome-Aware Oracle Definitions and Full Decomposition

All oracles are restricted to the already generated bounded action set plus baseline. Utility Oracle maximizes official Joint F1. Answer-Preserving Oracle first requires official Answer F1 no lower than baseline. Available-Opportunity Oracle selects the highest-Joint action among actions positive under the original training-positive title-utility criterion. Ties prefer baseline. Target-query outcomes are used only in this post-hoc analysis.

| Split | System | Answer F1 | SP F1 | Joint F1 | Delta Joint | Intervention |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development (nested 1,000) | baseline | 0.6114 | 0.4920 | 0.3241 | +0.0000 | 0.0% |
| Development (nested 1,000) | policy | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 26.0% |
| Development (nested 1,000) | utility_oracle | 0.7542 | 0.5536 | 0.4405 | +0.1164 | 39.2% |
| Development (nested 1,000) | answer_preserving_oracle | 0.7549 | 0.5531 | 0.4404 | +0.1163 | 39.2% |
| Development (nested 1,000) | available_opportunity_oracle | 0.7547 | 0.5271 | 0.4180 | +0.0939 | 39.2% |
| Original holdout (3,000) | baseline | 0.6183 | 0.4930 | 0.3292 | +0.0000 | 0.0% |
| Original holdout (3,000) | policy | 0.6271 | 0.4987 | 0.3356 | +0.0064 | 25.8% |
| Original holdout (3,000) | utility_oracle | 0.7686 | 0.5484 | 0.4398 | +0.1107 | 40.0% |
| Original holdout (3,000) | answer_preserving_oracle | 0.7688 | 0.5480 | 0.4397 | +0.1105 | 40.0% |
| Original holdout (3,000) | available_opportunity_oracle | 0.7161 | 0.5236 | 0.3911 | +0.0619 | 40.0% |
| Revision holdout (3,405) | baseline | 0.6129 | 0.4862 | 0.3201 | +0.0000 | 0.0% |
| Revision holdout (3,405) | policy | 0.6244 | 0.4923 | 0.3280 | +0.0080 | 25.9% |
| Revision holdout (3,405) | utility_oracle | 0.7510 | 0.5403 | 0.4253 | +0.1052 | 36.5% |
| Revision holdout (3,405) | answer_preserving_oracle | 0.7513 | 0.5399 | 0.4251 | +0.1051 | 36.5% |
| Revision holdout (3,405) | available_opportunity_oracle | 0.7096 | 0.5156 | 0.3830 | +0.0629 | 36.5% |

| Split | No training-positive opportunity | Opportunity missed | Positive selected | Harmful selected (Joint) | Mean regret | P90 | P95 | Zero regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development (nested 1,000) | 708 | 213 | 79 | 33 | 0.1099 | 0.4000 | 0.5000 | 63.7% |
| Original holdout (3,000) | 2316 | 465 | 219 | 115 | 0.1041 | 0.4000 | 0.5000 | 63.1% |
| Revision holdout (3,405) | 2638 | 515 | 252 | 125 | 0.0971 | 0.4000 | 0.5000 | 66.2% |

The holdout rows are post-hoc outcome-aware diagnostics. They do not prove generalization, validate holdout significance, or imply that the policy can approach oracle performance.

## R. Independent CrossEncoder-Top5 Details

Development official Joint F1 is 0.3300 for score order and 0.2404 for baseline-stable order; `ce_score_order` is frozen for both holdouts.

| Split | Variant | Answer F1 | SP F1 | Joint F1 | Joint vs baseline 95% CI | p | Joint vs Full 95% CI | p |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| development1000 | ce_score_order | 0.6052 | 0.5110 | 0.3300 | [-0.0109, +0.0222] | 0.4680 | [-0.0168, +0.0154] | 0.9668 |
| development1000 | ce_baseline_stable | 0.5855 | 0.3778 | 0.2404 | [-0.1016, -0.0659] | 0.0000 | not primary variant | -- |
| holdout3000 | ce_score_order | 0.6078 | 0.5240 | 0.3420 | [+0.0029, +0.0225] | 0.0116 | [-0.0033, +0.0156] | 0.1884 |
| holdout3000 | ce_baseline_stable | 0.5993 | 0.3739 | 0.2438 | [-0.0959, -0.0753] | 0.0000 | not primary variant | -- |
| revision3405 | ce_score_order | 0.6063 | 0.5220 | 0.3405 | [+0.0109, +0.0296] | 0.0000 | [+0.0034, +0.0211] | 0.0068 |
| revision3405 | ce_baseline_stable | 0.5968 | 0.3717 | 0.2410 | [-0.0890, -0.0689] | 0.0000 | not primary variant | -- |

Direct CrossEncoder-Top5 latency uses 50 warmup and 500 measured batch-one queries. Mean/median/P95 end-to-end latency is 149.90/135.47/262.59 ms, with context-match audit 100.0%.

## S. Development-Only Pair-Pruning Sensitivity

| k | Opportunity | Positive density | Policy coverage | Answer F1 | SP F1 | Joint F1 | Joint delta | Answer-drop | Generator ms | Total ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.0% | 13.2% | 26.0% | 0.6227 | 0.4976 | 0.3297 | +0.0056 | 1.4% | 68.19 | 211.63 |
| 2 | 28.6% | 14.0% | 26.0% | 0.6227 | 0.4967 | 0.3289 | +0.0048 | 1.3% | 68.39 | 211.83 |
| 3 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 68.60 | 212.04 |
| 5 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 69.01 | 212.45 |
| 7 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 69.43 | 212.86 |
| 10 | 29.2% | 14.7% | 26.0% | 0.6247 | 0.4973 | 0.3305 | +0.0064 | 1.3% | 70.05 | 213.48 |

No k is promoted. The quality rows are development-only and latency is component-scaled; k>3 cannot alter the frozen constructor's three pair-action slots.

## T. 2Wiki Structural Analysis

| Type | N | Baseline A/SP/J | Full delta A/SP/J | Joint 95% CI | raw p | BH-FDR p | opportunity | policy coverage |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| bridge_comparison | 252 | 0.6107/0.5037/0.3065 | +0.0010/-0.0011/+0.0004 | [-0.0109, +0.0119] | 0.9268 | 0.9872 | 48.0% | 24.6% |
| comparison | 252 | 0.6814/0.6973/0.4775 | +0.0099/-0.0048/+0.0001 | [-0.0187, +0.0183] | 0.9872 | 0.9872 | 14.3% | 25.8% |
| compositional | 382 | 0.2851/0.3158/0.1011 | +0.0179/+0.0013/+0.0087 | [+0.0007, +0.0174] | 0.0340 | 0.1360 | 34.6% | 26.4% |
| inference | 114 | 0.3192/0.2739/0.0888 | -0.0090/+0.0031/-0.0015 | [-0.0077, +0.0040] | 0.6460 | 0.9872 | 24.6% | 28.1% |

## Interpretation

No reasoning-type subgroup retains a statistically resolved Joint effect after BH-FDR correction. The available taxonomy therefore does not by itself explain the aggregate transfer uncertainty.

The distribution-shift table compares Hotpot-3,000 and 2Wiki on question/document length, pool size, entity overlap, bridge frequency, pair complementarity, policy scores, and action-family frequencies. Associations with transfer behavior are descriptive and do not establish causality.

Case studies are provided in `outputs/2wiki_analysis/case_studies.md`. The analysis uses official type labels, reports zero unmapped queries, controls the Joint subgroup family by BH-FDR, and makes no cross-domain success claim.

## U. Selected Intervention Profile

| Outcome | Feature | Logistic coefficient | OR per SD | Spearman rho | BH-FDR p |
| --- | --- | ---: | ---: | ---: | ---: |
| answer_gain | pair_complementarity_mean | +0.367 | 1.443 | +0.091 | 0.0008 |
| answer_gain | baseline_cross_encoder_mean | -0.365 | 0.694 | -0.134 | 0.0000 |
| answer_gain | candidate_cross_encoder_max | +0.359 | 1.432 | +0.073 | 0.0080 |
| answer_gain | no_intervention_needed | -0.292 | 0.747 | -0.101 | 0.0002 |
| answer_gain | added_doc_semantic_mean | +0.205 | 1.228 | +0.033 | 0.3140 |
| joint_gain | candidate_cross_encoder_max | +0.328 | 1.388 | +0.136 | 0.0000 |
| joint_gain | position_displacement | +0.316 | 1.371 | -0.051 | 0.0798 |
| joint_gain | baseline_cross_encoder_mean | -0.307 | 0.736 | -0.152 | 0.0000 |
| joint_gain | baseline_bm25_mean | -0.307 | 0.736 | -0.095 | 0.0005 |
| joint_gain | added_doc_opportunity_mean | +0.273 | 1.314 | +0.096 | 0.0004 |

## Interpretation boundary

These coefficients describe associations within the policy-selected subset. Selection changes the feature distribution, correlated predictors make coefficients non-causal, and multiple comparisons are controlled only for the Spearman family. The profile is not used to propose a new selector in this submission cycle.

## V. CrossEncoder-Full Disagreement Details

This is a post-hoc descriptive mechanism analysis over frozen per-query reader and official-metric outputs. It does not retrain either system, tune a threshold, or establish a causal effect of anchor preservation.

## CrossEncoder versus Frozen Top-5

| Split | Metric | Wins | Losses | Ties |
| --- | --- | ---: | ---: | ---: |
| Original holdout (3,000) | answer_f1 | 312 | 359 | 2329 |
| Original holdout (3,000) | sp_f1 | 850 | 760 | 1390 |
| Original holdout (3,000) | joint_f1 | 774 | 742 | 1484 |
| Revision holdout (3,405) | answer_f1 | 371 | 409 | 2625 |
| Revision holdout (3,405) | sp_f1 | 986 | 829 | 1590 |
| Revision holdout (3,405) | joint_f1 | 912 | 806 | 1687 |

## Full versus CrossEncoder

| Split | Metric | Wins | Losses | Ties |
| --- | --- | ---: | ---: | ---: |
| Original holdout (3,000) | answer_f1 | 338 | 263 | 2399 |
| Original holdout (3,000) | sp_f1 | 748 | 817 | 1435 |
| Original holdout (3,000) | joint_f1 | 726 | 738 | 1536 |
| Revision holdout (3,405) | answer_f1 | 398 | 322 | 2685 |
| Revision holdout (3,405) | sp_f1 | 829 | 942 | 1634 |
| Revision holdout (3,405) | joint_f1 | 800 | 875 | 1730 |

## Cross-events

| Split | Event | N | Proportion | Bootstrap 95% CI |
| --- | --- | ---: | ---: | --- |
| Original holdout (3,000) | CE SP up, Answer down | 63 | 2.1% | [1.6%, 2.6%] |
| Original holdout (3,000) | CE Joint up, Answer down | 5 | 0.2% | [0.0%, 0.3%] |
| Original holdout (3,000) | Full Answer up, CE Answer down | 1 | 0.0% | [0.0%, 0.1%] |
| Original holdout (3,000) | Both Answer up | 72 | 2.4% | [1.9%, 3.0%] |
| Original holdout (3,000) | Both Answer down | 39 | 1.3% | [0.9%, 1.7%] |
| Original holdout (3,000) | Both Joint up | 102 | 3.4% | [2.8%, 4.1%] |
| Revision holdout (3,405) | CE SP up, Answer down | 74 | 2.2% | [1.7%, 2.7%] |
| Revision holdout (3,405) | CE Joint up, Answer down | 9 | 0.3% | [0.1%, 0.4%] |
| Revision holdout (3,405) | Full Answer up, CE Answer down | 0 | 0.0% | [0.0%, 0.0%] |
| Revision holdout (3,405) | Both Answer up | 88 | 2.6% | [2.1%, 3.1%] |
| Revision holdout (3,405) | Both Answer down | 38 | 1.1% | [0.8%, 1.5%] |
| Revision holdout (3,405) | Both Joint up | 127 | 3.7% | [3.1%, 4.4%] |

## Full action families when Full improves Answer and CE lowers Answer

| Split | Action family | N | Share within event |
| --- | --- | ---: | ---: |
| Original holdout (3,000) | answer_anchor_first_reorder | 1 | 100.0% |

## Anchor-label boundary

The frozen artifacts contain inference-time anchor proxies but no reliable explicit answer-anchor label. We therefore do not report anchor retention or create an outcome-derived anchor label. The paired disagreement patterns are associations between system outputs, not causal proof of any particular document mechanism.

## W. Answer-Joint-Latency Figure Data

The figure uses only frozen points available for each split: RECOMP-660 is shown only for the original holdout and Lite only for the revision holdout. Non-dominance is assessed only among the evaluated systems, metrics, and measured post-retrieval latency. Source values are in `outputs/answer_joint_latency_points.csv`.

