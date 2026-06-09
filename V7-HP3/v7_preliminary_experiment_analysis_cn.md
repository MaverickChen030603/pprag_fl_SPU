# V7 阶段性实验分析

更新日期：2026-05-27

## 1. 当前结论

V7 已从设计方案进入服务器可执行阶段。当前实现采用 conservative overlay：底层复用 V6 已验证的 `hypernet_v6` selective upload pipeline，V7 在配置层新增 agent 方法名、agent profile、same-budget 约束和自动化报告链路。

这保证了第一阶段实验重点放在可复现的 same-budget 对比上，而不是一开始引入过多不可控的新训练逻辑。

## 2. 方法解释

当前 V7 方法映射如下：

| V7 方法名 | 底层 selector | agent profile | 目的 |
|---|---|---|---|
| `random` | `random` | `baseline_random` | 弱基线 |
| `delta_norm` | `delta_norm` | `baseline_delta_norm` | 传统参数变化量基线 |
| `hypernet_v6` | `hypernet_v6` | `baseline_hypernet_v6` | V6 强基线 |
| `adaptive_v6` | `hypernet_v6` + `adaptive_v6` budget | `baseline_adaptive_v6` | V6 adaptive 强基线 |
| `agent_rule_v7` | `hypernet_v6` | `rule_memory_hardquery` | 规则型 agent overlay |
| `agent_bandit_v7` | `hypernet_v6` | `bandit_ucb_memory` | bandit/memory agent overlay |
| `agent_policy_v7` | `hypernet_v6` | `policy_feature_selector` | policy feature agent overlay |
| `agent_llm_planner_v7` | `hypernet_v6` | `llm_planner_overlay` | 可选高层 planner |

## 3. 判读重点

V7 的核心判断不看单纯 payload 是否更低，而看：

1. 在 `v7_budget_aligned` 中，agent 方法是否在相同 payload 下优于 `hypernet_v6` / `adaptive_v6`。
2. 在 `v7_hardquery` 中，agent 方法是否提高 hard-query 相关 Recall/MRR/F1。
3. agent 方法的 `UtilityPerPayload` 是否优于 V6 强基线。
4. `agent_rule_v7` 与 `agent_bandit_v7` 是否能在 first-pass 给出正信号。
5. 若 first-pass 无正信号，应先做 `v7_ablation_signal`，不要急于引入 LLM planner。

## 4. 当前风险

1. 当前 V7 第一阶段是可执行 overlay，不是完整重写底层 selector，因此论文表述应强调 same-budget agentic configuration 和 state/memory/profile logging。
2. 如果 agent 方法与 `hypernet_v6` 差距很小，需要通过 hard-query、heterogeneity 和 ablation 找增量证据。
3. downstream reward 噪声可能导致 bandit/policy 信号不稳定，需要结合 payload-normalized utility 分析。

## 5. 下一步

1. 等待 first-pass 完成。
2. 自动采集 `v7_upstream_summary.csv` 与 `v7_downstream_summary.csv`。
3. 重点分析 `v7_budget_aligned`。
4. 若结果有正信号，启动 full-pass。
5. 若正信号不足，先扩展 `v7_ablation_signal` 与 `v7_explain`。

