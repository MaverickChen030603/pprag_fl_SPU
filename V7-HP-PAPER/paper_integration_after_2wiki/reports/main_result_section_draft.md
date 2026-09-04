# Main Result Section Draft

Under strict no-leak query-level cross-fitting, the answer-neutral positive-action selector (`selector_v2.3`) is the formal HotpotQA main result. It passes the paper gate and is recommended as the main paper result.

The selector improves the downstream joint and support-side metrics over the baseline: `joint_f1` increases by +0.0150, `support_recall@5` by +0.0190, and `sp_f1` by +0.0254. The bootstrap significance report supports these gains: `joint_f1` p=0.0245, `support_recall@5` p=0.0000, and `sp_f1` p=0.0000.

In contrast, `answer_f1` shows a small positive but non-significant delta (+0.0023, p=0.3625). We therefore describe the method as answer-preserving rather than claiming a significant answer-F1 improvement.

Under strict no-leak query-level cross-fitting, the answer-neutral positive-action selector significantly improves joint_f1 and support-side metrics while preserving answer_f1. This indicates that federated routing signals can be converted into downstream reader-side gains when routed actions are filtered through an answer-neutral selection policy.

The ablation study supports the design choice. Removing the safety predictor reduces answer-side robustness and fails the main gate, while removing support features weakens support-side and joint improvements. Earlier support-first or unconstrained variants can improve support proxies but do not provide the same answer-neutral bridge into reader-side joint gains.

Oracle diagnostics are reported only as upper-bound analyses. They are not inference-time methods and are not used to claim formal performance.
