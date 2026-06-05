# V7 下一阶段优化记录

更新时间：2026-05-29

## 优化目标

第一轮 `v7_budget_aligned` 与 `v7_hardquery` 的正式分析显示：agent 方法尚不能直接宣称成功。主要问题是：

- `agent_policy_v7` 使用 layerwise budget 后 payload 约为 `0.3163`，高于目标 `0.2249`。
- `v7_hardquery` 下游指标全为 1.0，baseline 过强，无法观察 agent 是否改善 hard-query。
- full pipeline finalizer/collector 曾默认读取 `pprag_fl_v6` / `rag_eval_all_v6` 或 `v7_adhoc` 路径，正式报告存在路径错配风险。

## 已执行优化

1. 修正 `agent_policy_v7` 默认预算控制：
   - 移除 `agent_policy_v7` / `agent_llm_planner_v7` 默认 `layerwise_budget=True`。
   - 保留 ablation/explain 中显式 layerwise 配置，用于诊断而不是主结论。
   - 目标是让 `agent_policy_v7` 在 budget-aligned/next suites 中严格对齐约 `0.2249` payload。

2. 新增强 hard-query suite：
   - suite 名称：`v7_hardquery_strong`
   - `hard_query_scale=4.0`
   - `hard_client_threshold=0.50`
   - `hard_client_bonus_topk=0`
   - `hard_budget_only=True`
   - 目标是放大 hard-query 排序差异，同时禁止预算扩张。

3. 启动下一阶段信号搜索：
   - `v7_heterogeneity`：72 runs
   - `v7_ablation_signal`：48 runs
   - `v7_hardquery_strong`：15 runs
   - 合计：135 upstream + 135 downstream

4. 修复正式 pipeline 路径：
   - upstream root 指向 `V7/outputs/pprag_fl_v7/<suite>`
   - downstream root 指向 `V7/outputs/rag_eval_all_v7/<suite>`
   - `all_v7` 汇总读取 `pprag_fl_v7` 与 `rag_eval_all_v7` 总目录。

5. 修复 collector：
   - 支持识别 `pprag_fl_v7` / `rag_eval_all_v7` 新目录结构。
   - 从 run 目录名和 `upstream_config.json` 中恢复 method、seed、topk、suite、agent profile。
   - 补齐 payload、hard-query scale、history window 与下游 `cos_1/cos_3/recall_3` 字段。

## 当前判断标准

下一阶段不以“全指标 1.0”为成功，而重点看：

- 同 payload 下 agent 是否优于 `adaptive_v6` / `hypernet_v6`。
- heterogeneity 中 rare-domain / hard-client 场景是否出现 agent 正向差异。
- ablation 中 `no_memory`、`no_hard_query`、`no_client_rarity` 是否改变选择行为和下游结果。
- `agent_policy_v7` 修正后 payload 是否回到约 `0.2249`。

