# V7-HP-PAPER selector_v2.2_scale_calibrated_budget 当前实验报告

生成时间：2026-06-19

## 1. 实验目的

`selector_v2.2_scale_calibrated_budget` 的目标是在不重跑 reader 的前提下，复用 `selector_v2_1` 已完成的 1000 条 raw candidate reader outputs，修复 `selector_v2.1` 从 100-sample gate 迁移到 final_1000 时出现的规模校准失败。

`selector_v2.1` 的主要失败形态是 final_1000 fallback 过高，实际选择仅 80/1000，导致最终指标几乎退化为 baseline，所有 delta 为 0。因此 v2.2 重点做三件事：

1. 审计候选 action 是否真正改变 reader context；
2. 过滤 ineffective action，并构建 query-action 级有效动作表；
3. 用 query-level 5-fold cross-fit 做 scale-calibrated budget calibration，避免在同一 1000 样本上直接调参再报告。

## 2. 实验配置

实验路径：

```text
/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/selector_v2_2/
```

复用输入：

```text
V7-HP-PAPER/selector_v2_1/outputs/final_1000/raw_candidate_eval/predictor_v2/candidate_rows.json
V7-HP-PAPER/selector_v2_1/outputs/final_1000/raw_candidate_eval/predictor_v2/predictor_predictions.json
V7-HP-PAPER/selector_v2_1/outputs/final_1000/final_1000_summary.json
V7-HP-PAPER/selector_v2_1/outputs/final_1000/per_example_delta.jsonl
V7-HP-PAPER/selector_v2_1/outputs/final_1000/significance_report.json
```

本轮没有重跑 reader，没有覆盖 `selector_v2_1`，并保持 strict no-leak：inference-time 不使用 gold support、gold answer、当前 query 的 reader outcome 或 oracle delta。Outcome 只用于 cross-fit 训练折内的配置校准，最终结果聚合 held-out fold 预测。

## 3. 已生成产物

关键产物均已生成：

```text
outputs/audit/action_audit_summary.json
outputs/audit/action_audit_rows.jsonl
outputs/action_table/action_table_summary.json
outputs/action_table/effective_action_table.jsonl
outputs/cv_calibration/cv_threshold_budget_summary.json
outputs/final_1000/final_1000_crossfit_summary.json
outputs/final_1000/per_example_delta.jsonl
outputs/final_1000/significance_report.json
outputs/ablation/ablation_summary.json
outputs/diagnostics/failure_summary.json
outputs/diagnostics/oracle_gap_summary.json
reports/v7_hp_paper_selector_v2_2_report.md
```

## 4. Action Audit 结果

候选动作层面：

```text
total_queries = 1000
total_candidate_actions = 5000
available_actions = 5000
effective_actions = 4517
ineffective_actions = 483
effective_action_rate = 0.9034
```

按候选族的 effective rate：

```text
top4_bg1 = 0.8090
insert1 = 0.9655
insert2 = 0.9940
bridge = 0.7830
```

候选族平均增量显示，`bridge` 是最强候选族：

```text
bridge:
  answer_f1_delta = +0.0015
  joint_f1_delta = +0.0286
  support_recall_delta = +0.0300
  sp_f1_delta = +0.0411

top4_bg1:
  answer_f1_delta = +0.0024
  joint_f1_delta = +0.0156
  support_recall_delta = +0.0150
  sp_f1_delta = +0.0197

insert1:
  answer_f1_delta = -0.0017
  joint_f1_delta = +0.0088
  support_recall_delta = +0.0130
  sp_f1_delta = +0.0162

insert2:
  answer_f1_delta = -0.0058
  joint_f1_delta = -0.0143
  support_recall_delta = -0.0180
  sp_f1_delta = -0.0244
```

结论：v2.2 的输入候选池本身是有效的；问题不再是“候选没有改变 context”，而是如何在 1000-scale 下选择足够多、且不伤 answer 的正向 action。

## 5. Final 1000 Cross-Fit 结果

v2.2 的 5-fold cross-fit 最终配置在 5 个 fold 上一致选择：

```text
candidate_family = insert1_plus_bridge
utility = support_first
selected_fraction = 0.5
safe_threshold = 0.55
risk_penalty_weight = 0.0
support_gain_threshold = null
```

最终 1000 条结果：

```text
selected_count = 500 / 1000
fallback_rate = 0.5000
effective_selected_count = 500
selected_effective_action_rate = 1.0000
```

候选选择分布：

```text
baseline_fallback = 500
keep_top3_insert1_slot4 = 296
keep_top3_bridge_insert1 = 168
keep_top2_insert1_slot3 = 36
```

指标对比：

| metric | baseline | selector_v2.2 | delta |
|---|---:|---:|---:|
| answer_access@k | 0.8330 | 0.8310 | -0.0020 |
| support_recall@k | 0.8190 | 0.8265 | +0.0075 |
| sp_f1 | 0.7483 | 0.7586 | +0.0103 |
| answer_em | 0.4800 | 0.4810 | +0.0010 |
| answer_f1 | 0.6100 | 0.6099 | -0.0001 |
| joint_f1 | 0.5170 | 0.5251 | +0.0081 |

严格 gate 结论：

```text
gate_pass = false
```

原因是 `answer_f1_delta = -0.000072`，虽然几乎为 0，但仍是负数。按照论文主结果标准，v2.2 不能直接宣称通过。

## 6. 显著性分析

paired bootstrap 结果：

| metric | mean_delta | 95% CI | p-value |
|---|---:|---:|---:|
| answer_f1 | -0.0001 | [-0.0105, +0.0099] | 0.4900 |
| joint_f1 | +0.0081 | [-0.0038, +0.0200] | 0.0985 |
| support_recall@5 | +0.0075 | [-0.0005, +0.0155] | 0.0400 |
| sp_f1 | +0.0103 | [-0.0004, +0.0213] | 0.0315 |

解释：

1. `support_recall@5` 与 `sp_f1` 出现稳定正信号，且 p-value 达到约 0.03-0.04；
2. `joint_f1` 有正向趋势，但 p-value 约 0.0985，尚不足以作为强显著主结论；
3. `answer_f1` 基本中性，微弱负增量不显著，但触发 strict gate 失败。

## 7. Failure Diagnosis

失败类型统计：

```text
n_cases = 1000
n_failure_cases = 971
candidate_pool_no_positive_action = 377
wrong_action_selected = 423
positive_action_rejected_by_budget = 119
under_abstention_answer_drop = 23
support_gain_no_reader_gain = 12
over_abstention = 4
answer_gain_no_support_gain = 13
```

关键诊断：

1. `ineffective_action_selected_count = 0`，说明 v2.2 已经解决了 ineffective action 选择问题；
2. `wrong_action_selected = 423` 是最大瓶颈，说明排序/打分仍然不能稳定命中真正带来 reader gain 的 action；
3. `fallback_but_positive_action_exists_count = 119`，说明 budget 或 safety 约束仍然过保守；
4. `selected_but_answer_drop_count = 23`，说明仍存在少量 support 增强但 answer 表述线索受损的样本。

## 8. Oracle Gap 诊断

由于 v2.2 strict gate 未通过，已生成 oracle gap 诊断：

```text
queries_with_actions = 996
oracle_best_joint_delta = +0.1411
oracle_best_answer_safe_joint_delta = +0.1508
oracle_support_delta = +0.0954
positive_candidate_rate = 0.2239
answer_safe_positive_candidate_rate = 0.2239
selector_recall_of_positive_candidates = 0.1839
```

解释：

候选池仍然存在很强的上界：如果能选到 oracle best 或 answer-safe positive action，joint_f1 理论上可提升约 +0.14 到 +0.15。这说明机制潜力仍在，当前瓶颈主要是 selector 对 positive candidates 的 recall 过低，而不是候选生成或 reader 本身完全无效。

## 9. 当前结论

v2.2 相比 v2.1 有实质进展：

1. fallback 从 0.92 降至 0.50；
2. selected_count 从 80 提升到 500；
3. selected action 100% effective；
4. support_recall 与 sp_f1 出现正向且接近显著/显著的提升；
5. joint_f1 有 +0.0081 的正向趋势。

但 v2.2 仍不能作为论文主结果：

1. strict gate 未通过；
2. answer_f1 微弱负增量；
3. joint_f1 尚未达到强显著；
4. selector 对 oracle positive candidates 的召回仅 0.1839。

论文表述建议：

`selector_v2.2` 可以作为“scale-calibrated selector 修复 v2.1 final_1000 退化”的中间证据，说明 support-side routing signal 在 1000-scale 下可恢复；但不能作为最终主方法结果。后续主线应转向 answer-neutral positive action selection，即提高 positive candidate recall，同时约束 answer_f1 不下降。

## 10. 下一步建议

建议启动 `selector_v2.3_answer_neutral_oracle_distill`：

1. 以 v2.2 的 effective action table 为训练集；
2. 学习区分 `positive_action` 与 `answer_drop_action`；
3. 目标函数从 support-first 改成 answer-neutral joint-first：

```text
maximize predicted_joint_gain
subject to predicted_answer_delta >= 0
and context_changed = true
```

4. cross-fit 中显式加入 answer lower-bound constraint，而不是只用 `safe_threshold`；
5. 提高 selector 对 oracle positive candidates 的召回，目标从当前 0.1839 提升到 0.30-0.40。

如果 v2.3 能在保持 `answer_f1_delta >= 0` 的同时保留 v2.2 的 `sp_f1` 正信号，就可以重新进入论文主结果候选。
