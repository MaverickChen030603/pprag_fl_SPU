# Experiment Section Draft

## Experimental Setup

We evaluate the paper-facing V7-HP-PAPER selector pipeline on a 1000-query HotpotQA validation subset. The final method, selector_v2.3, is trained and calibrated with query-level cross-fitting. No held-out query outcome is used at inference time, and no additional large-scale reader evaluation is run during paper finalization.

## Baselines and Variants

The comparison includes the baseline reader context, scale-calibrated selector_v2.2, and the final answer-neutral positive selector_v2.3. Oracle rows are treated strictly as diagnostic upper bounds and are not valid inference-time methods.

## Main Results

selector_v2.3 improves joint_f1 by 0.0150, support_recall@5 by 0.0190, and sp_f1 by 0.0254. answer_f1 changes by 0.0023; this is a small positive but not statistically significant change, so we describe the method as answer-preserving rather than answer-improving.

## Ablation Study

The ablation evidence indicates that two-stage and pairwise scoring drive the final cross-fit result. The answer-drop rejector alone is insufficient, and variants that weaken effective-action or answer-neutral constraints fail to match the final method.

## Candidate Pool and Oracle Gap

The candidate pool remains a major bottleneck: many queries do not contain any paper-positive action. This limits maximum recall and explains why the selector cannot benefit all examples even when it improves positive-action recall over selector_v2.2.

## Failure Analysis

Failures are dominated by candidate_pool_no_positive_action and positive_action_available_but_not_selected. The first points to candidate generation limits; the second suggests future work on more expressive but still no-leak ranking models.

## Discussion and Limitations

Client-side federated routing exposes support-relevant contexts, but naive insertion can hurt reader answer quality. The answer-neutral positive selector addresses this policy-action-to-reader gap by selecting only actions predicted to preserve answer quality while improving joint evidence utility. Under strict no-leak query-level cross-fitting, v2.3 achieves significant gains in joint_f1 and support-side metrics while preserving answer_f1.
