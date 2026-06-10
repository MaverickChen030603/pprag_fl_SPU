# V7-agent HotpotQA Agent-Rule 自动实验分析报告

生成时间：2026-06-10T14:39:30

## 数据集与目的

- 数据集切换为 HotpotQA fullwiki 派生的 `FedE/select_data_hotpot_train_5000.json`，保留 question、answer、supporting_titles 与 supporting context。
- 目的：放弃当前对方法不敏感的旧数据设置，用多跳证据链、rare bridge client 与 hard-query 场景放大 agent memory / hard-query / rarity signal 的作用。
- 当前 V7-agent/HP1 strict 指标是选择行为诊断；它不是 Hotpot 官方 answer F1/EM 或 supporting fact F1 的最终替代。

## 结论摘要

- `agent_rule_v7` 相对最佳 baseline 的 `hp1_multihop_score` gap = +0.0939，方向：正信号。
- `agent_bandit_v7` 相对最佳 baseline 的 `hp1_multihop_score` gap = +0.0214，方向：正信号。

- V7-agent 已收集 strict-eval 记录 12 条，平均预算 top-k：3.0000 ± 0.0000。
- 判断正信号时必须同步看预算：若 agent 预算显著高于 baseline，需要以 `hp1_budget_aligned` 为主结论。

## Upstream 完成数

- `hp1_budget_aligned`: 12 run

## 全局 strict 指标

| method/profile | n | avg_budget_topk_hp1 | bridge_block_recall_hp1 | early_evidence_recall_hp1 | selection_diversity_hp1 | hp1_multihop_score |
|---|---:|---:|---:|---:|---:|---:|
| hypernet_v6 | 3 | 3.0000 ± 0.0000 | 0.5842 ± 0.0091 | 0.0000 ± 0.0000 | 0.4333 ± 0.0471 | 0.4145 ± 0.0128 |
| adaptive_v6 | 3 | 3.0000 ± 0.0000 | 0.5842 ± 0.0091 | 0.0000 ± 0.0000 | 0.4333 ± 0.0471 | 0.4145 ± 0.0128 |
| agent_rule_v7 | 3 | 3.0000 ± 0.0000 | 0.5976 ± 0.0017 | 0.2000 ± 0.0000 | 0.5667 ± 0.0471 | 0.5084 ± 0.0088 |
| agent_bandit_v7 | 3 | 3.0000 ± 0.0000 | 0.6048 ± 0.0120 | 0.0000 ± 0.0000 | 0.5000 ± 0.0816 | 0.4359 ± 0.0170 |

## Suite: hp1_budget_aligned

| method/profile | n | avg_budget_topk_hp1 | bridge_block_recall_hp1 | early_evidence_recall_hp1 | selection_diversity_hp1 | hp1_multihop_score |
|---|---:|---:|---:|---:|---:|---:|
| hypernet_v6 | 3 | 3.0000 ± 0.0000 | 0.5842 ± 0.0091 | 0.0000 ± 0.0000 | 0.4333 ± 0.0471 | 0.4145 ± 0.0128 |
| adaptive_v6 | 3 | 3.0000 ± 0.0000 | 0.5842 ± 0.0091 | 0.0000 ± 0.0000 | 0.4333 ± 0.0471 | 0.4145 ± 0.0128 |
| agent_rule_v7 | 3 | 3.0000 ± 0.0000 | 0.5976 ± 0.0017 | 0.2000 ± 0.0000 | 0.5667 ± 0.0471 | 0.5084 ± 0.0088 |
| agent_bandit_v7 | 3 | 3.0000 ± 0.0000 | 0.6048 ± 0.0120 | 0.0000 ± 0.0000 | 0.5000 ± 0.0816 | 0.4359 ± 0.0170 |

- `agent_rule_v7` 相对最佳 baseline 的 `hp1_multihop_score` gap = +0.0939，方向：正信号。
- `agent_bandit_v7` 相对最佳 baseline 的 `hp1_multihop_score` gap = +0.0214，方向：正信号。

## Ablation: profile 拆分

- ablation 尚无结果。

## 下一步建议

1. 先看 `hp1_budget_aligned`，确认 agent gap 是否在相同 top-k 预算下仍存在。
2. 若 `hp1_rare_bridge_tail` gap 最大，把 rarity signal 固化进 agent policy，而不是只作为诊断标签。
3. 若 strict 指标有正信号，再接 Hotpot 官方 answer/supporting-fact 评估，避免只证明选择器会变而没有证明问答收益。
