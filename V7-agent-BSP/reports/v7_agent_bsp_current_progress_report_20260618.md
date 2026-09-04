# V7-agent-BSP 当前进展报告

生成时间：2026-06-18 18:50:50 JST

## 1. 执行状态

V7-agent-BSP 主流程已经完成，当前处于 reader sensitivity 后处理阶段。

| 项目 | 当前状态 |
|---|---:|
| 训练 final_artifacts | 105/105 |
| `v7bsp_main` | 50/50 |
| `v7bsp_bsp_methods` | 30/30 |
| `v7bsp_memory_ablation` | 25/25 |
| strict diagnostic | 105/105 |
| true FiD/T5 official eval | 105/105 |
| reader sensitivity eval | 152 个已完成，仍在运行 |

运行进程：

- 主流程 `runs/v7bsp_all.pid` 已结束。
- reader sensitivity 接力进程仍在运行：`runs/v7bsp_reader_sensitivity.pid`。
- 当前未见 hard error、OOM、Traceback。

## 2. Same-Budget 检查

Strict diagnostic 中所有方法均保持：`avg_topk = 3.0`，`budget_std = 0.0`。

Official eval 的 `avg_topk` 字段当前没有从 run metadata 正确带出，显示为 NaN；因此预算判断以 strict diagnostic 的 `avg_topk/budget_std` 为准。该问题需要后续修复 collector metadata，但不影响本轮 strict same-budget 结论。

## 3. Strict Diagnostic 初步结果

| method | n | bridge_block_recall_hp1 | early_evidence_recall_hp1 | target_block_recall_hp1 | selection_diversity_hp1 | hp1_multihop_score | avg_topk | budget_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent_pm_bandit_slot | 5 | 0.429091 | 0.363636 | 0.396364 | 0.84 | 0.548073 | 3 | 0 |
| agent_bsp_memory_bandit_no_rarity_state | 5 | 0.412364 | 0.381091 | 0.396727 | 0.84 | 0.54696 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 10 | 0.411636 | 0.381818 | 0.396727 | 0.84 | 0.546902 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 10 | 0.411636 | 0.381818 | 0.396727 | 0.84 | 0.546902 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 15 | 0.411636 | 0.381818 | 0.396727 | 0.84 | 0.546902 | 3 | 0 |
| agent_bsp_memory_bandit_no_instability_state | 5 | 0.368727 | 0.381818 | 0.375273 | 0.9 | 0.542167 | 3 | 0 |
| agent_pm_dynamic_no_memory | 5 | 0.402182 | 0.345455 | 0.373818 | 0.86 | 0.535942 | 3 | 0 |
| agent_bsp_memory_bandit_no_failure_state | 5 | 0.373818 | 0.363636 | 0.368727 | 0.88 | 0.534516 | 3 | 0 |
| agent_bsp_bandit_retrieval | 5 | 0.386909 | 0.345455 | 0.366182 | 0.88 | 0.533985 | 3 | 0 |
| agent_bsp_bandit_strict | 5 | 0.386909 | 0.345455 | 0.366182 | 0.88 | 0.533985 | 3 | 0 |
| agent_bsp_bandit_reader | 5 | 0.386909 | 0.345455 | 0.366182 | 0.88 | 0.533985 | 3 | 0 |
| agent_rule_v7_dynamic | 5 | 0.436364 | 0.363636 | 0.4 | 0.7 | 0.522909 | 3 | 0 |
| agent_pm_dynamic_full | 5 | 0.432727 | 0.363636 | 0.398182 | 0.64 | 0.509491 | 3 | 0 |
| agent_rule_v7 | 5 | 0.598545 | 0.2 | 0.399273 | 0.54 | 0.503433 | 3 | 0 |
| agent_bsp_memory_bandit_no_history_state | 5 | 0.536 | 0.2 | 0.368 | 0.46 | 0.46304 | 3 | 0 |
| adaptive_v6 | 5 | 0.588364 | 0 | 0.294182 | 0.42 | 0.413462 | 3 | 0 |
| hypernet_v6 | 5 | 0.588364 | 0 | 0.294182 | 0.42 | 0.413462 | 3 | 0 |

当前 strict 结论：

1. `agent_pm_bandit_slot` 仍是最强 strict 方法，HP1 = 0.5481。
2. BSP memory bandit 三个主方法非常接近，HP1 = 0.5469，略低于 `agent_pm_bandit_slot` 约 0.00117。
3. BSP 明显超过 `agent_rule_v7_dynamic`，后者 HP1 = 0.5229。
4. 无 history state 明显失败，HP1 = 0.4630，说明 history/memory state 对 BSP 有强正贡献。
5. 去掉 instability state 反而略高；去掉 rarity state 与 full 几乎相同，说明当前 rarity/instability 设计仍需再校准。

## 4. Memory State Ablation

| method | hp1_multihop_score | avg_topk | budget_std |
| --- | --- | --- | --- |
| agent_bsp_memory_bandit_no_failure_state | 0.534516 | 3 | 0 |
| agent_bsp_memory_bandit_no_history_state | 0.46304 | 3 | 0 |
| agent_bsp_memory_bandit_no_instability_state | 0.542167 | 3 | 0 |
| agent_bsp_memory_bandit_no_rarity_state | 0.54696 | 3 | 0 |
| agent_bsp_memory_bandit_reader | 0.546902 | 3 | 0 |
| agent_bsp_memory_bandit_retrieval | 0.546902 | 3 | 0 |
| agent_bsp_memory_bandit_strict | 0.546902 | 3 | 0 |

关键判断：

- `agent_bsp_memory_bandit_retrieval` vs `no_history_state`：+0.08386 HP1，p = 0.00062，history state 是明确正信号。
- `agent_bsp_memory_bandit_retrieval` vs `no_failure_state`：+0.01239 HP1，p = 0.00058，failure state 有正贡献。
- `agent_bsp_memory_bandit_retrieval` vs `no_rarity_state`：-0.000058，p = 0.083，rarity state 当前不是稳定正贡献。
- `no_instability_state` 高于 full，说明 instability penalty 可能仍偏保守。

## 5. Method-Balanced True FiD/T5

| method | answer_F1 | support_F1 | joint_F1 | support_title_recall | budget_std |
| --- | --- | --- | --- | --- | --- |
| hypernet_v6 | 0.656796 | 0.506 | 0.33216 | 0.7437 | 0 |
| agent_bsp_memory_bandit_reader | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0 |
| agent_bsp_memory_bandit_retrieval | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0 |
| agent_bsp_memory_bandit_strict | 0.654558 | 0.506 | 0.331841 | 0.7445 | 0 |
| agent_pm_bandit_slot | 0.654558 | 0.5062 | 0.331841 | 0.7445 | 0 |
| agent_pm_dynamic_full | 0.654598 | 0.5065 | 0.331841 | 0.7445 | 0 |
| agent_rule_v7 | 0.654658 | 0.50625 | 0.331741 | 0.74425 | 0 |
| agent_pm_dynamic_no_memory | 0.654598 | 0.5059 | 0.331521 | 0.7443 | 0 |

Official eval 结论：

- true FiD/T5 已完成 105/105，均为 `t5-base`，beam=3，max_input_length=768。
- 端到端指标仍被压平，joint F1 大约集中在 0.3318 到 0.3324。
- `hypernet_v6` 的 official joint F1 反而略高于 BSP/agent 方法，但差距极小。
- BSP 的 strict 改善没有明显传导到 official QA endpoint。

## 6. Reader Sensitivity 当前进展

reader sensitivity 已完成 152 个 eval，仍在继续。当前已刷新进分析表的部分结果如下：

| method | beam_size | max_input_length | passage_ordering | n | answer_EM | answer_F1 | support_EM | support_F1 | joint_EM | joint_F1 | support_title_recall | budget_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent_bsp_memory_bandit_reader | 1 | 512 | agent_priority | 10 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_reader | 1 | 512 | gold_oracle_debug | 10 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_reader | 1 | 512 | retrieval_score | 10 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_reader | 1 | 768 | retrieval_score | 8 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_retrieval | 1 | 512 | agent_priority | 15 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_retrieval | 1 | 512 | gold_oracle_debug | 15 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_retrieval | 1 | 512 | retrieval_score | 15 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_retrieval | 1 | 768 | retrieval_score | 5 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_strict | 1 | 512 | agent_priority | 10 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_strict | 1 | 512 | gold_oracle_debug | 10 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_strict | 1 | 512 | retrieval_score | 10 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_bsp_memory_bandit_strict | 1 | 768 | retrieval_score | 5 | 0.56 | 0.63511 | 0.193333 | 0.505 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_pm_bandit_slot | 1 | 512 | agent_priority | 5 | 0.56 | 0.63511 | 0.194667 | 0.505667 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_pm_bandit_slot | 1 | 512 | gold_oracle_debug | 5 | 0.56 | 0.63511 | 0.194667 | 0.505667 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_pm_bandit_slot | 1 | 512 | retrieval_score | 5 | 0.56 | 0.63511 | 0.194667 | 0.505667 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_rule_v7 | 1 | 512 | agent_priority | 5 | 0.56 | 0.63511 | 0.196667 | 0.506667 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_rule_v7 | 1 | 512 | gold_oracle_debug | 5 | 0.56 | 0.63511 | 0.196667 | 0.506667 | 0.106667 | 0.319551 | 0.731667 | 0 |
| agent_rule_v7 | 1 | 512 | retrieval_score | 5 | 0.56 | 0.63511 | 0.196667 | 0.506667 | 0.106667 | 0.319551 | 0.731667 | 0 |

当前 sensitivity 早期观察：

- beam=1、max_input_length=512/768 的部分组合中，不同 method 的 answer F1/joint F1 几乎完全一致。
- `retrieval_score`、`agent_priority`、`gold_oracle_debug` 当前也没有拉开差距。
- 这提示 official reader 侧目前对 selection 差异不敏感，或者 passage ordering 诊断还没有真正改变 reader 输入排序，需要后续核查 `gold_oracle_debug` 的实现是否足够强。

## 7. 统计检验

| method_a | method_b | metric | n | mean_delta | wilcoxon_p | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agent_pm_bandit_slot | agent_rule_v7_dynamic | hp1_multihop_score | 5 | 0.0251636 | 0.0625 | 0.0251636 | 0.0251636 |
| agent_bsp_memory_bandit_strict | agent_pm_bandit_slot | hp1_multihop_score | 10 | -0.00117091 | 0.00195312 | -0.00117091 | -0.00117091 |
| agent_bsp_memory_bandit_retrieval | agent_pm_bandit_slot | hp1_multihop_score | 15 | -0.00117091 | 0.000287414 | -0.00117091 | -0.00117091 |
| agent_bsp_memory_bandit_reader | agent_pm_bandit_slot | hp1_multihop_score | 10 | -0.00117091 | 0.00195312 | -0.00117091 | -0.00117091 |
| agent_bsp_memory_bandit_reader | agent_pm_dynamic_full | joint_F1 | 10 | 0 |  | 0 | 0 |
| agent_bsp_memory_bandit_retrieval | agent_bsp_memory_bandit_no_failure_state | hp1_multihop_score | 15 | 0.0123855 | 0.000580579 | 0.0123855 | 0.0123855 |
| agent_bsp_memory_bandit_retrieval | agent_bsp_memory_bandit_no_rarity_state | hp1_multihop_score | 15 | -5.81818e-05 | 0.0832645 | -5.81818e-05 | -5.81818e-05 |
| agent_bsp_memory_bandit_retrieval | agent_bsp_memory_bandit_no_history_state | hp1_multihop_score | 15 | 0.0838618 | 0.000622548 | 0.0838618 | 0.0838618 |

当前判断：

- `agent_pm_bandit_slot` vs `agent_rule_v7_dynamic`：+0.02516 HP1，p=0.0625，5 seeds 下接近显著。
- BSP memory bandit full vs `agent_pm_bandit_slot`：约 -0.00117 HP1，统计上稳定低一点，因此不能宣称 BSP full 超过 bandit slot。
- BSP memory retrieval vs no-history：+0.08386 HP1，p=0.00062，是本轮最强正信号之一。
- BSP memory retrieval vs no-failure：+0.01239 HP1，p=0.00058，failure state 有贡献。

## 8. 当前成功等级判断

- A 级：暂未达到。Official joint F1 未超过 `agent_pm_bandit_slot`，hard-query/rare-domain 真子集仍需等完整 sensitivity 和更强 per-query alignment。
- B 级：部分达到。Memory state 中 history/failure state 有稳定正贡献；BSP 在 strict 上稳定改变 upload behavior，但没有超过 `agent_pm_bandit_slot`。
- C 级：基本达到。same-budget 下 BSP 改变通信结构，dynamic/bandit planning 继续优于普通 dynamic rule，PM full 失败原因进一步定位为 memory 直接加权不如 slot-level state planning，而 official reader 不敏感仍是主要瓶颈。

## 9. 当前结论

不要直接宣称 V7-agent-BSP 成功超过所有方法。更准确的当前结论是：

> Under strict same-budget top-k=3, BSP memory-bandit planning preserves the strong slot-planning signal and clearly benefits from history/failure state, but it does not yet outperform the previous `agent_pm_bandit_slot` strict baseline. True FiD/T5 metrics remain largely insensitive, so the main positive evidence is currently behavioral/diagnostic rather than endpoint QA improvement.

## 10. 待办

1. 等 reader sensitivity 全网格完成。
2. 重新生成 `analysis/reader_sensitivity_summary.csv` 和完整报告。
3. 修复 official eval metadata，让 `avg_topk=3.0` 写入 official balanced 表。
4. 强化 `gold_oracle_debug` passage ordering，确认它是否真的改变 reader 输入。
5. 做更精确的 per-query selection trace 对齐，把 `selected_blocks/slot_allocation/bandit_action` 接进 official per-query QA。
6. 若继续优化方法，应以 `agent_pm_bandit_slot` 为强基线，重点改 BSP 的 rarity/instability state，而不是再加重 block score memory。

## 11. 文件位置

- 主日志：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/runs/v7bsp_all.nohup.log`
- reader sensitivity 日志：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/runs/v7bsp_reader_sensitivity.nohup.log`
- 当前报告：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/reports/v7_agent_bsp_current_progress_report_20260618.md`
- 完整报告模板/自动报告：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP/reports/v7_agent_bsp_complete_report_20260617.md`
