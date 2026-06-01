# V6-H1 完整实验分析

- 生成时间：2026-06-01T14:17:16

## 1. 实验问题

V6-H1 针对 V3-V6 下游差距不明显的问题，专门构建稳定 hard-query 子集。其目标不是继续扩大通信预算，而是在更有判别力的查询集合上检验选择性上传策略是否真正改善检索排序质量。

## 2. 主要观察

- 当前尚未发现 full-pipeline 报告，说明自动化流程可能仍在运行或尚未进入报告阶段。

## 3. 后续判断标准

- 优先比较 `v6h1_budget_aligned_stable_hardquery` 中 `hypernet_v6` 与 `hypernet_v3/random/delta_norm` 的同预算下游指标。
- 其次查看 `v6h1_heterogeneity_stable_hardquery`，判断强异构场景下 hard-query 子集是否放大方法差异。
- 最后结合 `hard_queries/stable_hard_queries.json` 的数量与原因分布，确认 hard-query 子集是否足够难且不是偶然噪声。
