# Response to Major-Revision Concerns

## 1. Marginal Absolute Gains

**Concern.** The average improvements are small.

**Response.** We agree. We now report the frozen 3,000-query population effect together with direct paired effects on selected and fallback queries. The 774 selected interventions have Answer/SP/Joint deltas +0.0340/+0.0219/+0.0250, while fallbacks are unchanged. We also add deployment cost so the conditional effect is not presented without its operational trade-off. The Abstract, Results Sec. 5.1, and Cost Sec. 6 use "modest" rather than impact-inflating language.

## 2. Limited Domain Transfer

**Concern.** Frozen 2Wiki results do not establish generalization and show elevated answer risk.

**Response.** We agree. The zero-shot result remains visible and is labeled non-significant with flat support and 6.92% selected answer-drop. We add K-shot calibration from 2Wiki train under five seeds without retraining the generator or reading evaluation outcomes. The experiment is complete, but no setting reaches the pre-specified <=4% answer-drop target. The paper has separate Zero-Shot Frozen Transfer and Few-Shot Safety Calibration subsections, and safety transfer remains an explicit limitation.

## 3. Mixed Semantic Component Ablations

**Concern.** The prior narrative implied that a complex set of semantic components all contributed consistently.

**Response.** We agree. We build fully nested Lite-Lexical-Pair, Lite-Semantic-Pair, and PairChain variants. Lite-Lexical-Pair reaches Answer/SP/Joint F1 0.6183/0.4922/0.3290 on development, then is frozen before a 3,405-query revision holdout. On that untouched split its Joint difference from Full is -0.0063, failing both point and CI non-inferiority. We therefore retain Full as the primary implementation. Pair complementarity remains the clearest learned mechanism, chains the structural mechanism, and anchors/safety the risk controls; other semantic features are a joint implementation recipe rather than individually validated contributions.

## 4. Unfair RECOMP Comparison

**Concern.** Top-1 RECOMP retained 47.1 tokens versus approximately 660 for our method.

**Response.** We agree. The previous Top-1 condition is no longer evidence of general superiority. We add RECOMP budgets 64/128/256/384/512/660 and source-order Baseline-Truncated controls under the same Top-5 input, FLAN reader, decoding, support predictor, and paired evaluation. At the fixed matched point, RECOMP uses 637.8 tokens and obtains Answer/SP/Joint F1 0.6049/0.4704/0.3082. The budget is frozen before the 3,000-query run. The revised Sec. 5.3 discusses context-construction objectives, not universal superiority.

## 5. High Operational Complexity

**Concern.** The method's small gains may not justify its development and deployment cost.

**Response.** We agree that the previous cost evidence was incomplete. Candidate reader outcomes are generated offline; deployment runs the answer reader once on the final context. We report action counts, stored labels, static encoder/cross-encoder/pair calls, context tokens, final-reader latency percentiles, memory, and throughput. The Lite variant removes costly modules but fails the independent non-inferiority test, so it is not promoted as the main method. A comparable end-to-end generator latency harness and historical GPU-hour manifest are unavailable and remain `[NEEDS MEASUREMENT]`/`[NOT AVAILABLE]`. Sec. 6 narrows deployment to bounded, auditable post-retrieval QA and retains operational complexity as review risk.

## Remaining Limitations

The gains remain small at population level; strict Lite non-inferiority may remain uncertain; zero-shot transfer is weak; and the fixed per-query pool does not test corpus-scale retrieval. We preserve these limits rather than treating the revision as a universal solution.
