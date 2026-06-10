# V7-agent Budget-Aligned 结果解读与论文分析

生成时间：2026-06-10T14:42:43

## 一句话结论

V7-agent 的 `agent_rule_v7` 在严格 same-budget top-k=3 下，成功把 HP1 的 early-evidence 选择从 baseline 的 0.0000 拉到 0.2000，并将 `hp1_multihop_score` 从 0.4145 提升到 0.5084；这说明 agent policy 的 hard-query/early-evidence 对齐确实改变了选择行为，并产生了较强的诊断正信号。

## 实验设置

- 数据：HotpotQA fullwiki 派生 `FedE/select_data_hotpot_train_5000.json`。
- Suite：`hp1_budget_aligned`。
- Methods：`hypernet_v6`, `adaptive_v6`, `agent_rule_v7`, `agent_bandit_v7`。
- Seeds：0, 1, 2。
- 预算协议：所有方法 `avg_budget_topk_hp1 = 3.0000 ± 0.0000`。
- 指标性质：当前为 strict diagnostic selection metrics，不是 Hotpot 官方 answer/supporting-fact F1/EM。

## 主结果表

| method | n | top-k | bridge | early | target | diversity | hp1 score |
|---|---:|---:|---:|---:|---:|---:|---:|
| hypernet_v6 | 3 | 3.0000 | 0.5842 | 0.0000 | 0.2921 | 0.4333 | 0.4145 |
| adaptive_v6 | 3 | 3.0000 | 0.5842 | 0.0000 | 0.2921 | 0.4333 | 0.4145 |
| agent_bandit_v7 | 3 | 3.0000 | 0.6048 | 0.0000 | 0.3024 | 0.5000 | 0.4359 |
| agent_rule_v7 | 3 | 3.0000 | 0.5976 | 0.2000 | 0.3988 | 0.5667 | 0.5084 |

## 相对 baseline 的增益

- `agent_rule_v7`: `hp1_multihop_score` delta = +0.0939 (+22.6%).
- `agent_rule_v7`: `early_evidence_recall_hp1` delta = +0.2000.
- `agent_rule_v7`: `target_block_recall_hp1` delta = +0.1067 (+36.5%).
- `agent_bandit_v7`: `hp1_multihop_score` delta = +0.0214 (+5.2%).
- `agent_bandit_v7`: `early_evidence_recall_hp1` delta = +0.0000.
- `agent_bandit_v7`: `target_block_recall_hp1` delta = +0.0103 (+3.5%).

## 逐 seed 稳定性

| seed | baseline score | agent_rule score | score gap | agent_rule early |
|---:|---:|---:|---:|---:|
| 0 | 0.4069 | 0.5146 | +0.1077 | 0.2000 |
| 1 | 0.4041 | 0.5146 | +0.1105 | 0.2000 |
| 2 | 0.4326 | 0.4960 | +0.0634 | 0.2000 |

## 选择行为解释

- `hypernet_v6`: events=165, early=0 (0.0%), bridge=482 (77.6%), other=139 (22.4%); top blocks=[('pooler', 165), ('encoder.layer.8', 160), ('encoder.layer.11', 153), ('encoder.layer.7', 139), ('encoder.layer.9', 4)].
- `adaptive_v6`: events=165, early=0 (0.0%), bridge=482 (77.6%), other=139 (22.4%); top blocks=[('pooler', 165), ('encoder.layer.8', 160), ('encoder.layer.11', 153), ('encoder.layer.7', 139), ('encoder.layer.9', 4)].
- `agent_bandit_v7`: events=165, early=0 (0.0%), bridge=499 (76.9%), other=150 (23.1%); top blocks=[('pooler', 165), ('encoder.layer.8', 163), ('encoder.layer.11', 159), ('encoder.layer.7', 149), ('encoder.layer.9', 12)].
- `agent_rule_v7`: events=165, early=165 (25.0%), bridge=493 (74.7%), other=2 (0.3%); top blocks=[('encoder.layer.8', 165), ('pooler', 165), ('encoder.layer.11', 163), ('encoder.layer.0', 154), ('encoder.layer.3', 11)].

解读：baseline 与 bandit 主要集中在 high-layer bridge blocks，early evidence 仍为 0；`agent_rule_v7` 通过 hard-query focused scoring 和 early-evidence coverage，在每个 post-warmup event 的 top-3 中稳定放入一个 early evidence block，同时保留 bridge block，因此 target recall 与 overall multihop score 同时上升。

## 论文表述建议

可以写成：Under an identical Top-K communication budget, the proposed rule-based client agent substantially improves diagnostic multihop evidence coverage. In particular, it recovers early-evidence block recall from 0.0000 to 0.2000 while preserving bridge-block recall, leading to a +22.6% relative gain in the HP1 multihop diagnostic score over the strongest V6 baseline.

中文表达：在严格相同通信预算下，agent_rule_v7 不依赖额外 payload，而是通过局部记忆、hard-query 对齐和 early-evidence coverage 改变 block 选择结构，使多跳证据链中的低层证据块开始进入上传集合。

## 谨慎边界

- 当前结论证明的是选择器对 Hotpot 多跳结构更敏感，而不是最终 QA 效果已经提升。
- 样本量为 3 seeds，适合作为强诊断信号，但还需要扩展到 rare/hard/full suite。
- `agent_bandit_v7` 提升较小，且 early evidence 仍为 0；论文主线应暂时放在 `agent_rule_v7`。
- 下一步必须接 Hotpot official EM/F1/supporting-fact F1，确认 selection gain 是否传导到下游 RAG。

## 归档内容

- 自动报告：`实验分析报告/V7-agent/v7_agent_auto_analysis_20260610_143930.md`
- strict CSV：`V7-agent/outputs/hp1_strict_eval/hp1_budget_aligned/hp1_strict_summary.csv`
- 归档目录：`实验分析报告/V7-agent/archive_v7_agent_budget_aligned_20260610`
- manifest：`实验分析报告/V7-agent/archive_v7_agent_budget_aligned_20260610/archive_manifest.json`
