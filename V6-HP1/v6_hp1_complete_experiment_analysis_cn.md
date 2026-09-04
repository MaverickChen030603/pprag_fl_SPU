# V6-HP1 完整实验分析

- 生成时间：2026-06-10T02:20:02

## 1. 实验问题

V6-HP1 直接放弃了原本区分度较低的数据集，改用 HotpotQA fullwiki。目标是在更难、更多跳推理、更容易拉开排序差异的下游任务上，重新检验 same-budget 选择性上传是否真的比 V3/V5/V6 更值。

## 2. 主要观察

- 已收集 8 个 full-pipeline 报告。
- 重点先看 `v6hp1_budget_aligned`：如果 Hotpot 上同预算差异被放大，它会是最关键的证据。
- 其次看 `v6hp1_heterogeneity`：如果在更强 non-IID 下 V6-HP1 比 `random/delta_norm` 更省且下游更稳，则能补足主结果差异不明显的问题。
- `v6hp1_hardquery` 会进一步验证 harder query 上同预算策略是否优于启发式基线。

## 3. 判断标准

- 优先比较 `v6hp1_budget_aligned` 中 `hypernet_v6` 与 `hypernet_v3/random/delta_norm` 的 `recall_3/mrr/NDCG`。
- 若 payload 接近而下游指标更高，则说明 Hotpot 已经成功放大方法差异。
- 若 Hotpot 上仍然几乎没有差距，则主要瓶颈更可能来自上游训练信号本身，而不只是旧数据集太简单。
