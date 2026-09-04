# V7-HP-PAPER 完整实验分析报告

生成时间：2026-06-19  
主结果版本：`selector_v2.3_answer_neutral_positive_selector`  
证据包目录：`V7-HP-PAPER/paper_finalization/`

## 1. 执行摘要

V7-HP-PAPER 系列的核心问题是：客户端 federated routing 已经能够暴露更多 support-relevant contexts，但简单把这些 context 插入 reader 输入并不必然带来下游 QA 收益，甚至可能损伤 answer quality。该现象在前几轮实验中表现为 support 指标有正信号，而 answer_f1 不稳定或微弱下降。

最终的 `selector_v2.3` 采用 answer-neutral positive-action selection，在 strict no-leak、query-level cross-fitting 条件下，将 routing-side support gain 转化为 reader-side joint_f1 gain，并保护 answer_f1 不下降。

最终结论：

- `selector_v2.3` 应冻结为当前论文主结果。
- 不建议继续 v2.4 盲目调参。
- 可以写入论文主 claim：joint_f1、support_recall@5、sp_f1 在 strict no-leak cross-fit 下显著提升。
- 需要谨慎表述：answer_f1 为小幅正向但不显著，因此应称为 answer-preserving，而不是 answer-improving。

## 2. 实验目的与科学问题

本实验要回答的问题不是“能否用 oracle 选出更好 context”，而是在严格无泄漏条件下，agent/routing 产生的候选 context 是否能被一个 inference-time 可用的 selector 转化为正式 QA 收益。

具体科学问题：

1. Federated routing 是否产生了真实可用的 support-side candidate？
2. 这些 candidate 是否能在不使用 gold answer/support 的情况下被选择？
3. selector 是否能避免 support gain 损伤 reader 的 answer quality？
4. 最终收益是否能体现在 HotpotQA 的 joint_f1 与 support metrics 上？
5. 当前系统的主要瓶颈在 routing、selector，还是 candidate generation？

## 3. 方法演进

### HP4 机制验证

HP4 证明 soft routing、hybrid retrieval 与 counterfactual credit assignment 可以打破早期 context flattening bottleneck。它说明 routing 有潜力改善 support exposure，但也暴露出 policy-action-to-reader gap：support 更好并不自动等于 reader answer 更好。

### v1 / v2

v1 使用早期 predictor selector，能够产生一定选择行为，但过于激进，answer 风险较大。v2 引入更强 abstention / safety 控制，能保护 answer，但 fallback 过高，导致收益被压平。

### v2.1

v2.1 的 budgeted risk relaxation 在 100-sample gate 上有效，但迁移到 final_1000 时失败。主要问题是 selected_count 太低，fallback 过高，final_1000 几乎退化成 baseline。

### v2.2

v2.2 通过 scale-calibrated budget 与 effective action filtering 修复了 v2.1 的规模退化问题。它让 selected action 全部有效，并恢复 support-side 正信号，但 answer_f1 仍为微弱负增量，因此 strict gate 未通过。

v2.2 final_1000：

| 指标 | delta |
|---|---:|
| answer_f1 | -0.0001 |
| joint_f1 | +0.0081 |
| support_recall@5 | +0.0075 |
| sp_f1 | +0.0103 |
| fallback_rate | 0.5000 |
| selected_effective_action_rate | 1.0000 |
| gate_pass | false |

### v2.3

v2.3 的关键变化是从 support-first selection 转向 answer-neutral positive selection。它不再只追求 support gain，而是优先选择预测为 answer-safe 且 joint-beneficial 的 action。

v2.3 使用 query-level 5-fold cross-fitting：

- train folds：训练 selector / 校准 threshold 与 budget；
- held-out fold：只使用模型预测与 no-leak features 选择 action；
- 汇总所有 held-out predictions 得到 final_1000。

## 4. 最终主结果

主结果表：

| 指标 | Baseline | v2.2 | v2.3 |
|---|---:|---:|---:|
| answer_access@5 | 0.8330 | 0.8310 | 0.8450 |
| support_recall@5 | 0.8190 | 0.8265 | 0.8380 |
| sp_f1 | 0.7483 | 0.7586 | 0.7737 |
| answer_em | 0.4800 | 0.4810 | 0.4810 |
| answer_f1 | 0.6100 | 0.6099 | 0.6122 |
| joint_f1 | 0.5170 | 0.5251 | 0.5320 |

v2.3 相对 baseline 的增量：

| 指标 | delta |
|---|---:|
| answer_access@5 | +0.0120 |
| support_recall@5 | +0.0190 |
| sp_f1 | +0.0254 |
| answer_em | +0.0010 |
| answer_f1 | +0.0023 |
| joint_f1 | +0.0150 |
| fallback_rate | 0.5000 |
| selected_effective_action_rate | 1.0000 |
| positive_candidate_recall | 0.3288 |
| gate_pass | true |
| paper_main_recommended | true |

解释：

v2.3 同时满足 answer_f1 非负、joint_f1 正、support_recall 正、sp_f1 正、effective action rate 为 1.0、fallback 不超过 0.80、positive candidate recall 超过 v2.2 reference 的 gate 条件。

## 5. 显著性分析

paired bootstrap 结果：

| 指标 | mean_delta | 95% CI | p-value |
|---|---:|---:|---:|
| answer_f1 | +0.0023 | [-0.0114, +0.0158] | 0.3625 |
| joint_f1 | +0.0150 | [+0.0001, +0.0302] | 0.0245 |
| support_recall@5 | +0.0190 | [+0.0085, +0.0295] | 0.0000 |
| sp_f1 | +0.0254 | [+0.0106, +0.0393] | 0.0000 |

论文表述边界：

- 可以说 joint_f1 显著提升；
- 可以说 support_recall@5 和 sp_f1 显著提升；
- 可以说 answer_f1 被保护，且出现小幅正向变化；
- 不应说 answer_f1 显著提升。

## 6. Ablation 分析

主要 ablation：

| 变体 | answer_f1_delta | joint_f1_delta | support_recall_delta | sp_f1_delta | positive_recall | gate |
|---|---:|---:|---:|---:|---:|---|
| v2.3 main | +0.0023 | +0.0150 | +0.0190 | +0.0254 | 0.3288 | true |
| two_stage | +0.0051 | +0.0113 | +0.0120 | +0.0150 | 0.2928 | true |
| paper_positive_classifier | +0.0017 | +0.0076 | +0.0120 | +0.0150 | 0.2838 | true |
| answer_drop_rejector_support_ranker | -0.0005 | +0.0078 | +0.0125 | +0.0166 | 0.1216 | false |
| constrained_regression | +0.0050 | +0.0092 | +0.0105 | +0.0150 | 0.2883 | false |
| no_answer_constraint | +0.0013 | +0.0133 | +0.0185 | +0.0247 | 0.3198 | true |
| no_support_features | +0.0002 | +0.0127 | +0.0135 | +0.0184 | 0.3063 | true |
| no_safety_predictor | -0.0070 | +0.0040 | +0.0110 | +0.0136 | 0.2658 | false |
| v2.2 support_first | -0.0001 | +0.0081 | +0.0075 | +0.0103 | n/a | false |

结论：

1. final cross-fit 的 two-stage / pairwise 混合配置在 joint_f1 和 paper-main criteria 上最强。
2. 单独 two-stage 也通过 gate，但 joint_f1 低于主配置。
3. answer-drop rejector alone 不足以解决问题，positive recall 过低。
4. 去掉 safety predictor 会导致 answer_f1 明显负增量，说明 safety/answer-neutral 信号是必要的。
5. 去掉 support features 仍有正信号，但幅度下降，说明 routing/support features 对最终收益有贡献。

## 7. Candidate Pool 质量与上限

candidate pool 统计：

| candidate_family | n_actions | paper_positive_rate | answer_drop_rate | joint_positive_rate |
|---|---:|---:|---:|---:|
| bridge | 1000 | 0.0940 | 0.0380 | 0.0960 |
| insert1 | 2000 | 0.1075 | 0.0635 | 0.1095 |
| insert2 | 1000 | 0.0690 | 0.0560 | 0.0710 |
| top4_bg1 | 1000 | 0.0700 | 0.0450 | 0.0710 |

总体：

- num_queries = 1000
- num_actions = 5000
- paper_positive_rate = 0.0896
- queries_with_no_positive_action = 778
- queries_with_at_least_one_positive_action = 222

解释：

778/1000 queries 没有 paper-positive action，这是当前系统最主要的上限。selector 无法在没有正向候选的 query 上创造收益，因此后续如果要继续提升，应优先改善 candidate generation，而不是继续调 selector。

## 8. Feature Importance

positive action 与 non-positive action 的区分特征中，较强信号包括：

| feature | positive_mean | non_positive_mean | effect_size |
|---|---:|---:|---:|
| num_added_docs | 1.0246 | 0.6848 | +0.6767 |
| num_removed_docs | 1.0246 | 0.6848 | +0.6767 |
| safe_answer_prob | 0.5398 | 0.6202 | -0.5883 |
| agent_weight_delta | 0.0050 | 0.0023 | +0.4196 |
| support_proxy_delta_vs_replaced_doc | 0.0319 | 0.0143 | +0.4092 |
| support_proxy_delta | 0.0312 | 0.0140 | +0.4057 |
| answer_risk_score | 0.2663 | 0.2110 | +0.3967 |
| prefix3_preserved | 0.1786 | 0.3464 | -0.3887 |
| title_bridge_score | 0.0469 | 0.0393 | +0.2687 |

解释：

1. `agent_weight_delta`、`support_proxy_delta`、`title_bridge_score` 有正区分度，说明 routing/support/sparse bridge features 对识别 positive action 有贡献。
2. `safe_answer_prob` 的关系不是简单越高越好，因为 positive action 常常需要替换 context，可能牺牲一部分保守安全分；这也是需要 answer-neutral selector 而不是单纯 safety gate 的原因。
3. `answer_risk_score` 可用于解释风险控制，但不能被解读为 answer 显著提升的证据。

## 9. No-Leak / Cross-Fit 审计

审计结果：

- query fold disjoint：已由 deterministic split 验证；
- held-out outcome 不用于 inference：已由 source review 验证；
- training labels 仅来自 train folds：已由 source review 验证；
- threshold / selected_fraction 仅在 train folds 校准：已由 source review 验证；
- oracle diagnostic 与 formal selector 分离：已验证；
- inference feature 不含 gold answer / gold support：已由 feature list 验证；
- fold_count = 5。

审计边界：

这是 artifact/source-level audit，不重跑 reader，也不检查隐藏外部状态。论文中应写为“we audit the artifacts and source path”而不是声称数学证明式无泄漏。

## 10. Failure Analysis

v2.3 failure summary：

| failure_label | count |
|---|---:|
| candidate_pool_no_positive_action | 778 |
| selected_positive | 73 |
| positive_action_available_but_not_selected | 102 |
| wrong_action_selected | 41 |
| answer_drop_selected | 2 |
| support_positive_but_joint_negative | 4 |

解释：

1. 最大失败来源是 candidate pool 没有 positive action；
2. 第二类失败是 positive action 存在但 selector 未选中，仍有 ranking 改进空间；
3. answer_drop_selected 很少，说明 answer-neutral 约束总体有效；
4. support_positive_but_joint_negative 说明 support improvement 不总是转化为 reader joint improvement。

## 11. Case Study 使用建议

已导出三类 case：

- success cases；
- answer-neutral cases；
- failure cases。

写论文时建议每类选 1 个代表案例：

1. success：展示 selector 插入 bridge/support evidence 后 joint_f1 提升；
2. answer-neutral：展示 answer_f1 维持不变但 support/joint 改善；
3. failure：展示 candidate_pool_no_positive_action 或 positive_action_available_but_not_selected，以说明限制。

如果 case 中显示 gold answer，应明确标注其仅用于 analysis display，没有作为 inference feature。

## 12. 论文 claim 边界

可以写：

1. The proposed selector improves joint_f1 significantly under strict no-leak cross-fitting.
2. It improves support_recall@5 and sp_f1 significantly.
3. It preserves answer_f1 with a small non-significant positive delta.
4. Positive-action recall improves substantially compared with v2.2.
5. Candidate pool quality remains the main bottleneck.

不应写：

1. It significantly improves answer_f1.
2. It solves reader sensitivity completely.
3. It reaches the oracle upper bound.
4. It works for all multi-hop QA cases.
5. Support gain always improves answer generation.

推荐贡献表述：

> Answer-neutral action selection for federated RAG routing.

或：

> Bridging routing-side support gains and reader-side joint QA gains under no-leak constraints.

## 13. 结论

V7-HP-PAPER 的最终科学叙事已经清楚：federated routing 能暴露 support-relevant candidates，但 naive support insertion 会遇到 reader-side answer risk。`selector_v2.3` 通过 answer-neutral positive-action selection，在 strict no-leak query-level cross-fitting 下，将 support-side routing signal 转化为显著 joint_f1/support-side gains，同时保护 answer_f1。

建议冻结 v2.3 作为论文主结果，进入论文撰写阶段。后续如需补实验，应只做低成本 robustness 或 case-level diagnostic，不建议继续 v2.4 selector 调参。
