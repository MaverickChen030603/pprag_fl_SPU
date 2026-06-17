# V7-agent-PM 当前进展与初步分析报告

生成日期：2026-06-17  
项目路径：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-PM`  
本地路径：`/Users/iilab/ForAgent/实验分析报告/V7-agent-PM/v7_agent_pm_current_progress_report_20260617.md`

## 1. 当前执行状态

V7-agent-PM 本轮全流程已经完成，不再是运行中状态。

| 模块 | 状态 | 完成数 |
|---|---:|---:|
| `v7pm_main` | 完成 | 25/25 |
| `v7pm_memory_ablation` | 完成 | 30/30 |
| `v7pm_dynamic_ablation` | 完成 | 40/40 |
| `v7pm_bandit_slot` | 完成 | 15/15 |
| 训练总 run | 完成 | 110/110 |
| strict diagnostic | 完成 | 4 个 suite |
| true FiD/T5 official eval | 完成 | 110/110 |

环境修复也已完成：

- `sentencepiece ok 0.2.1`
- `t5 tokenizer ok`
- official eval 日志显示：`[FiDReader] Model loaded.`

因此，本轮 official eval 不再是 V7-agent-2 中的 fallback reader，而是真正加载了 `t5-base` 的 FiD/T5 reader 路径。

## 2. 实验目的回顾

V7-agent-PM 的目标是在 V7-agent-2 基础上进一步验证：

1. 在 fixed same-budget top-k=3 下，Planning-Memory Agent 是否能超过当前最强的 `agent_rule_v7_dynamic`。
2. PM 的收益是否来自 memory、failure recovery、rarity 和 instability penalty，而不是单纯 early prior。
3. dynamic slot allocation 是否优于 fixed early slot heuristic。
4. true FiD/T5 official eval 是否能看到 joint F1、support F1 或 support title recall 的端到端提升。
5. 如果整体 official eval 不明显，strict diagnostic 或 subgroup/per-query 行为是否仍能证明 agentic upload planning 改变了 selection behavior。

## 3. Same-budget 约束确认

所有 strict diagnostic 汇总中的方法均保持：

- `avg_topk = 3.0`
- `budget_std = 0.0`

因此，本轮方法差异不能解释为隐性扩大通信预算。

## 4. Strict Diagnostic 主要结果

### 4.1 主线对比

| 方法 | n | early recall | bridge recall | diversity | HP1 score | avg top-k |
|---|---:|---:|---:|---:|---:|---:|
| `hypernet_v6` | 5 | 0.0000 | 0.5884 | 0.4200 | 0.4135 | 3.0 |
| `adaptive_v6` | 5 | 0.0000 | 0.5884 | 0.4200 | 0.4135 | 3.0 |
| `agent_rule_v7` | 5 | 0.2000 | 0.5985 | 0.5400 | 0.5034 | 3.0 |
| `agent_rule_v7_dynamic` | 10 | 0.3636 | 0.4364 | 0.7000 | 0.5229 | 3.0 |
| `agent_pm_dynamic_full` | 15 | 0.3636 | 0.4327 | 0.6400 | 0.5095 | 3.0 |
| `agent_pm_bandit_slot` | 5 | 0.3636 | 0.4291 | 0.8400 | 0.5481 | 3.0 |

初步判断：

- `agent_rule_v7_dynamic` 继续明显优于 baseline 与普通 `agent_rule_v7`。
- `agent_pm_dynamic_full` 没有超过 `agent_rule_v7_dynamic`，反而 HP1 score 低约 0.0134。
- `agent_pm_bandit_slot` 是 strict diagnostic 中最强方法，HP1 score 达到 0.5481，并且 diversity 最高，为 0.84。

### 4.2 统计检验

| 比较 | 指标 | n | mean delta | p-value | 解释 |
|---|---|---:|---:|---:|---|
| `agent_rule_v7_dynamic` vs `agent_pm_dynamic_full` | HP1 score | 30 | +0.0134 | 1.58e-05 | dynamic 显著高于 PM full |
| `agent_rule_v7_dynamic` vs `agent_pm_bandit_slot` | HP1 score | 10 | -0.0252 | 0.00195 | bandit slot 显著高于 dynamic |
| `agent_pm_dynamic_full` vs `agent_pm_dynamic_no_memory` | HP1 score | 15 | -0.0265 | 0.00207 | no-memory 显著高于 PM full |
| `agent_pm_dynamic_full` vs `agent_pm_dynamic_no_failure_memory` | HP1 score | 15 | -0.0081 | 0.1236 | 去掉 failure memory 后略高，但不显著 |
| `agent_dynamic_slot` vs `agent_fixed_slot_1` | HP1 score | 5 | +0.0192 | 0.0625 | dynamic 优于 fixed-1，接近显著 |
| `agent_dynamic_slot` vs `agent_fixed_slot_2` | HP1 score | 5 | +0.0178 | 0.0625 | dynamic 优于 fixed-2，接近显著 |

注意：表中的 mean delta 是前者减后者。负值表示后者更高。

## 5. Memory Ablation 结果

| 方法 | n | early recall | bridge recall | diversity | HP1 score |
|---|---:|---:|---:|---:|---:|
| `agent_pm_dynamic_full` | 15 | 0.3636 | 0.4327 | 0.6400 | 0.5095 |
| `agent_pm_dynamic_no_memory` | 5 | 0.3455 | 0.4022 | 0.8600 | 0.5359 |
| `agent_pm_dynamic_no_failure_memory` | 5 | 0.3636 | 0.3920 | 0.7600 | 0.5176 |
| `agent_pm_dynamic_no_rarity_memory` | 5 | 0.3455 | 0.4509 | 0.6200 | 0.5069 |
| `agent_pm_dynamic_no_instability_penalty` | 5 | 0.3636 | 0.4175 | 0.7000 | 0.5155 |
| `agent_pm_dynamic_no_utility_ema` | 5 | 0.3636 | 0.4327 | 0.6400 | 0.5095 |

判断：

- 本轮没有证明 PM full memory 产生正向独立贡献。
- 反而 `no_memory` 在 strict diagnostic 上高于 PM full，HP1 score 为 0.5359。
- `no_utility_ema` 与 PM full 完全一致，说明 utility EMA 当前没有实际影响。
- `no_instability_penalty` 略高于 PM full，说明当前 instability penalty 可能压制了必要探索。
- `no_failure_memory` 略高于 PM full但不显著，failure memory 目前没有稳定收益。

这说明 PM full 的 memory 设计还不够好：它确实进入了选择链路，但当前权重和 credit assignment 可能引入了噪声或过度保守。

## 6. Dynamic Planning Ablation

| 方法 | early recall | bridge recall | diversity | HP1 score |
|---|---:|---:|---:|---:|
| `agent_fixed_slot_0` | 0.0000 | 0.6313 | 0.5000 | 0.4462 |
| `agent_fixed_slot_1` | 0.2000 | 0.5956 | 0.4800 | 0.4903 |
| `agent_fixed_slot_2` | 0.4000 | 0.3993 | 0.5600 | 0.4917 |
| `agent_fixed_slot_3` | 0.6000 | 0.2000 | 0.5600 | 0.4760 |
| `agent_dynamic_slot` | 0.3636 | 0.4327 | 0.6400 | 0.5095 |
| `agent_dynamic_no_hardness` | 0.2000 | 0.5956 | 0.4800 | 0.4903 |
| `agent_dynamic_no_rarity` | 0.3455 | 0.4509 | 0.6200 | 0.5069 |
| `agent_dynamic_no_bridge_guard` | 0.3636 | 0.4327 | 0.6400 | 0.5095 |

判断：

- dynamic slot 明确优于 fixed slot heuristic，尤其优于 fixed-0/1/2/3 的 HP1 score。
- 去掉 hardness 后退化到 fixed-slot-1 水平，说明 hardness 判断是 dynamic planning 的核心。
- 去掉 rarity 后小幅下降，rarity 有弱正贡献。
- 去掉 bridge guard 没有变化，说明当前 bridge guard 实现尚未发挥作用。

## 7. Bandit Slot Policy

`agent_pm_bandit_slot` 的 strict diagnostic 最强：

| 方法 | early recall | bridge recall | diversity | HP1 score |
|---|---:|---:|---:|---:|
| `agent_rule_v7_dynamic` | 0.3636 | 0.4364 | 0.7000 | 0.5229 |
| `agent_pm_dynamic_full` | 0.3636 | 0.4327 | 0.6400 | 0.5095 |
| `agent_pm_bandit_slot` | 0.3636 | 0.4291 | 0.8400 | 0.5481 |

判断：

- 把 bandit action 从“直接选 block”改成“选择 early slot 数量”是有价值的。
- 它没有增加预算，仍为 top-k=3。
- 它的优势主要体现在 selection diversity 与 HP1 score，而不是 early recall 本身。

## 8. True FiD/T5 Official Eval

本轮 official eval 已确认加载 `t5-base`：

- 日志显示 `[FiDReader] Model loaded.`
- 不再是 fallback reader。

但端到端指标没有明显拉开：

| 方法 | answer F1 | sp F1 | joint F1 | support title recall |
|---|---:|---:|---:|---:|
| `hypernet_v6` | 0.6540 | 0.5060 | 0.3309 | 0.7437 |
| `agent_rule_v7` | 0.6537 | 0.5063 | 0.3315 | 0.7443 |
| `agent_rule_v7_dynamic` | 未单独聚合为官方表项 | - | - | - |
| `agent_pm_dynamic_full` | 0.6537 | 0.5065 | 0.3315 | 0.7445 |
| `agent_pm_bandit_slot` | 0.6536 | 0.5062 | 0.3315 | 0.7445 |
| `agent_pm_dynamic_no_memory` | 0.6531 | 0.5059 | 0.3309 | 0.7443 |

判断：

- true FiD/T5 eval 没有复现 strict diagnostic 中的明显差距。
- 大多数方法 joint F1 都集中在约 0.3310 到 0.3315。
- support title recall 也集中在约 0.7437 到 0.7445。
- 当前 official eval 更像是“检索/reader 端瓶颈把方法差异压平”，不能支持 PM full 端到端胜出。

## 9. 成功判据逐项判断

| 成功问题 | 当前判断 |
|---|---|
| `agent_pm_dynamic_full` 是否超过 `agent_rule_v7_dynamic` | 否。strict 上 PM full 低于 dynamic |
| PM 增益是否来自 memory/failure/rarity | 否。memory full 没有正贡献，no-memory 反而更高 |
| dynamic slot 是否优于 fixed slot heuristic | 是。strict 上 dynamic slot 优于 fixed slot，p=0.0625 接近显著 |
| true FiD/T5 official eval 是否有 joint F1/support F1 提升 | 否。整体差异极小 |
| hard-query/rare-domain/hard-client 是否稳定提升 | 目前 subgroup 是 strict proxy，不足以支撑真实子集结论 |
| per-query case 是否证明 planning 改变选择行为 | per-query 文件已产出，但 selection trace 与 QA case 的精确对齐仍需增强 |

## 10. 当前结论

V7-agent-PM 的结论不是“PM full 成功超过 dynamic”，而是：

1. `agent_rule_v7_dynamic` 仍是稳定强基线。
2. `agent_pm_dynamic_full` 当前设计没有超过 dynamic，memory 组件可能引入噪声或过度约束。
3. `agent_pm_bandit_slot` 在 strict diagnostic 上给出了最强正信号，说明“规划 early slot 数量”比“直接加 memory 权重”更有潜力。
4. Dynamic planning 明确优于 fixed early slot heuristic，hardness 是关键因素。
5. True FiD/T5 eval 已修复并完成，但端到端指标未明显拉开。

可用于论文的谨慎表述：

> Under a strict same-budget top-k=3 protocol, dynamic upload planning consistently changes block selection and improves multihop-oriented strict diagnostics over fixed-slot heuristics. However, the current Planning-Memory full policy does not outperform the dynamic rule baseline, and true FiD/T5 end-to-end QA metrics remain largely insensitive. The strongest signal comes from slot-level bandit planning, suggesting that future agentic FL policies should plan communication structure rather than only reweight block utilities.

## 11. 下一步建议

1. 把 `agent_pm_bandit_slot` 作为下一轮主线，而不是 `agent_pm_dynamic_full`。
2. 重写 memory：不要把 memory 直接加权进 score，而是用于 slot-level policy、hard-query failure clustering 和 delayed reward credit assignment。
3. 修复 subgroup analysis：目前是 strict proxy，需要用真实 query-level hard/easy、rare/common、hard-client 标注重算。
4. 强化 per-query behavior 对齐：把 selection trace 的 `selected_blocks`、`score_components`、`replacement_reason` 与 official per-query QA 结果连接起来。
5. 在 official eval 上减少重复 suite 聚合造成的 method n 不平衡，单独生成主对比表：baseline vs rule_dynamic vs pm_full vs pm_bandit_slot。
6. 检查 true FiD/T5 reader 的生成质量，目前 answer F1 低于 fallback 版本，可能需要调 beam、输入长度或 reader prompt。

## 12. 文件位置

服务器主报告：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-PM/reports/v7_agent_pm_complete_report_20260616.md`

服务器当前进展报告：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-PM/reports/v7_agent_pm_current_progress_report_20260617.md`

本地当前进展报告：

`/Users/iilab/ForAgent/实验分析报告/V7-agent-PM/v7_agent_pm_current_progress_report_20260617.md`

