# Review Accuracy Audit

This audit compares the review claims with the final frozen V5 artifacts. Categories are **accurate**, **partially accurate**, **outdated**, **overstated**, and **incorrect**.

| Review claim | Category | Frozen evidence | Final correction |
|---|---|---|---|
| Online latency is missing. | **Outdated** | Same-machine A100 timing now reports Frozen Top-5 140.88 ms/query, Full 213.48, Lite 143.97, and RECOMP-660 169.64; every system uses one final reader call. | The concern was valid for an earlier draft but is resolved in the current evidence package. |
| The same-source answer-drop rate is 2.0%. | **Outdated** for selected-policy risk | Direct selected-query accounting gives 60/774 = 7.75% on the 3,000 holdout and 69/881 = 7.83% on the 3,405 holdout. | The final paper uses 7.75%/7.83% whenever it discusses risk among selected interventions. An earlier aggregate/population quantity is not reused as selected answer-drop. |
| The intervention produces a significant gain on every selected query. | **Incorrect** | Original selected Answer wins/losses/ties are 89/60/625; Joint is 141/115/518. Revision selected Answer is 107/69/705; Joint is 169/125/587. Every median and interquartile endpoint is zero. | The conditional means are descriptive; most selected queries tie and some are harmed. |
| Larger selected-subset means prove a causal intervention effect. | **Incorrect** | Selection is policy-dependent and no randomized treatment assignment is used. | The paper separates population effects from descriptive effects conditional on the policy's chosen subset. |
| The RECOMP comparison proves general superiority. | **Overstated** | At about 660 tokens, RECOMP Joint F1 is 0.3259 versus 0.3292 for Frozen Top-5; delta -0.0033, 95% CI [-0.0109, +0.0044], p=0.4172. Full is 0.3356, but action spaces differ. | The result is a budget-controlled protocol comparison, not a universal ranking or an end-to-end reproduction claim. |
| The paper lacks an independent simplification test. | **Outdated** | Lite minus Full Joint F1 is -0.0063, 95% CI [-0.0104, -0.0023], against a frozen 0.002 non-inferiority margin. | The independent test exists and fails; the failure is retained. |
| The method has an external transfer success. | **Incorrect** | 2Wiki Answer/SP/Joint p-values are .1116/.6928/.3296, and the best few-shot answer-drop is 5.10% versus a 4% target. | The paper reports transfer as a failed diagnostic and unresolved limitation. |
| The system is evaluated at corpus scale. | **Incorrect** | The official distractor pool is about ten documents; 2,973/3,000 queries have at least ten, only one has twenty, and none has fifty or one hundred. | The paper states a bounded post-retrieval candidate-pool scope. |
| Two independent readers replicate the complete pipeline. | **Partially accurate** | FLAN and UnifiedQA have directionally positive Answer F1 deltas, but they share the same support predictor. | The second model is an answer-reader directional check; SP is not independently replicated and Joint contains a shared component. |
| The nested evaluation prevents holdout outcome selection. | **Accurate** | Generator and selector fit outer training folds, thresholds use inner out-of-fold predictions, and outer-test outcomes are not used for architecture or threshold selection. | Retained as a central methodological strength without claiming that nesting guarantees deployment safety. |

## Interpretation

The strongest resolved review concern is cost measurement. The strongest remaining concerns are the small population effect relative to a 1.52x latency ratio, nonzero selected-query harm, same-source and bounded-pool scope, shared support prediction in the second-reader check, and failed external transfer. None can be removed by prose; the revision makes each visible and narrows the claim accordingly.
