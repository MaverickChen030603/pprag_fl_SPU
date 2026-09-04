# Final Response to Review Concerns

We thank the reviewers for identifying weaknesses that required both additional controls and narrower claims. All new evaluations preserve the frozen Full system; neither the 3,000-query nor the 3,405-query holdout outcomes were used to select methods or thresholds.

## Weakness 1: Marginal Absolute Gains

**Response.** We agree that the population gains are modest and now state this in the Abstract, Introduction, Results, Limitations, and Conclusion. We add a second untouched 3,405-query confirmation and exact intervals and p-values for both holdouts. We also add direct policy-selected accounting: on the original 774 interventions, Answer/SP/Joint F1 change by +0.0340/+0.0219/+0.0250, with Answer wins/losses/ties of 89/60/625 and Joint 141/115/518. These are labeled descriptive conditional effects, not causal or arbitrary-query effects. The population deltas remain primary, and the new end-to-end cost table prevents a practical-impact claim beyond the evidence.

## Weakness 2: Limited Transfer

**Response.** We retain the failed frozen 2Wiki result: coverage is 26.0%, selected answer-drop is 6.92%, Answer and Joint changes are non-significant, and SP is flat. We add target-train gate calibration for K={16,32,64,128} under five seeds, with frozen generator, reader, prompt, action space, and evaluation outcomes. The best mean answer-drop is 5.10%, an improvement from 6.92% but above the pre-specified 4% success criterion. We therefore keep distribution-shift safety as an explicit limitation.

## Weakness 3: Mixed Semantic Ablations

**Response.** We agree that mixed component results do not support treating every semantic signal as a separate novelty claim. The paper now centers pair complementarity, bounded two-document chains, anchor preservation, and selective safety. We evaluate a development-frozen Lite simplification, but on the untouched revision holdout Lite minus Full Joint F1 is -0.0063 and fails the 0.002 non-inferiority rule. Full remains empirically stronger. Missing-hop, MPNet, cross-encoder, and document-opportunity components are described as the Full implementation recipe, not as independently proven monotonic contributions.

## Weakness 4: Unfair RECOMP Comparison

**Response.** We agree that the approximately 47-token Top-1 condition cannot support a general ranking against near-full contexts. We add an official-compressor 64-660-token development curve, a source-order Baseline-Truncated control, and a frozen 660-token protocol evaluated on the 3,000-query holdout with the same Top-5 input, FLAN reader, support predictor, and metric code. RECOMP-660 changes Joint F1 by -0.0033 ([-0.0109, +0.0044], p=0.4172). We remove broad superiority language and describe this as an official-compressor implementation under reader and budget adaptation. Matched tokens do not equate the structural action spaces.

## Weakness 5: Complexity

**Response.** We now separate offline outcome labeling and training from online inference. Online inference calls the answer reader once on the final context, not once per action. Under a shared 50-warmup/500-measurement protocol, Full measures 213.48 ms/query end to end after retrieval, including 70.05 ms generation, 0.61 ms selection, and 142.59 ms reading; Frozen Top-5 measures 140.88 ms. Lite reduces this to 143.97 ms but fails non-inferiority. We therefore report an explicit quality-cost trade-off and limit scope to bounded post-retrieval pools. Historical offline GPU-hour totals were not recorded and are therefore unavailable.
