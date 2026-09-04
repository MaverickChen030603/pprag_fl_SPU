# V7-HP-PAPER selector_v2.2 当前进展报告

查询时间：2026-06-19 15:40 JST  
实验目录：`/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/selector_v2_2/`

## 1. 当前运行状态

服务器 live check 显示：

```text
selector_v2_2 相关进程：无
pipeline 状态：已结束
日志状态：无 hard error；仅有 libtinfo version warning，不影响结果
```

`selector_v2_2` 的主要脚本、outputs 与 reports 均已存在。当前不需要继续等待，也没有 reader 任务仍在运行。

## 2. 已生成产物

核心产物已齐全：

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
reports/v7_hp_paper_selector_v2_2_status_report_20260619.md
```

报告也已镜像到：

```text
实验分析报告/V7-HP-PAPER/v7_hp_paper_selector_v2_2_status_report_latest.md
实验分析报告/V7-HP-PAPER/v7_hp_paper_selector_v2_2_report_latest.md
```

## 3. Final 1000 Cross-Fit 结果

v2.2 的 final_1000 结果如下：

| metric | baseline | selector_v2.2 | delta |
|---|---:|---:|---:|
| answer_access@k | 0.8330 | 0.8310 | -0.0020 |
| support_recall@k | 0.8190 | 0.8265 | +0.0075 |
| sp_f1 | 0.7483 | 0.7586 | +0.0103 |
| answer_em | 0.4800 | 0.4810 | +0.0010 |
| answer_f1 | 0.6100 | 0.6099 | -0.0001 |
| joint_f1 | 0.5170 | 0.5251 | +0.0081 |

选择行为：

```text
selected_count = 500 / 1000
fallback_rate = 0.5000
effective_selected_count = 500
selected_effective_action_rate = 1.0000
gate_pass = false
```

5 个 cross-fit fold 都选择同一配置：

```text
candidate_family = insert1_plus_bridge
utility = support_first
selected_fraction = 0.5
safe_threshold = 0.55
risk_penalty_weight = 0.0
support_gain_threshold = null
```

## 4. 显著性结果

paired bootstrap：

| metric | mean_delta | 95% CI | p-value |
|---|---:|---:|---:|
| answer_f1 | -0.0001 | [-0.0105, +0.0099] | 0.4900 |
| joint_f1 | +0.0081 | [-0.0038, +0.0200] | 0.0985 |
| support_recall@5 | +0.0075 | [-0.0005, +0.0155] | 0.0400 |
| sp_f1 | +0.0103 | [-0.0004, +0.0213] | 0.0315 |

解释：

`support_recall@5` 与 `sp_f1` 有正信号，并达到或接近统计显著；`joint_f1` 有正趋势但未达到强显著；`answer_f1` 基本中性但略负，因此严格 gate 失败。

## 5. Action Audit 与 Ablation 判断

Action audit：

```text
total_candidate_actions = 5000
effective_actions = 4517
ineffective_actions = 483
effective_action_rate = 0.9034
```

v2.2 选择出的 action：

```text
selected_effective_action_rate = 1.0000
```

这说明 v2.2 已修复 v2.1 final_1000 的一个关键问题：不再选择无效 context action。Ablation 也支持这一点：不做 effective filter 时，selected_effective_action_rate 只有 `0.0033`，几乎退化成 baseline。

## 6. Failure Diagnosis

主要失败来源：

```text
wrong_action_selected = 423
candidate_pool_no_positive_action = 377
positive_action_rejected_by_budget = 119
under_abstention_answer_drop = 23
support_gain_no_reader_gain = 12
```

关键解释：

1. 当前最大瓶颈不是 action 是否有效，而是 selector 是否选中真正带来 reader gain 的 positive action；
2. `positive_action_rejected_by_budget = 119` 表明仍有一批本可提升的 action 被 budget 或排序挡掉；
3. `under_abstention_answer_drop = 23` 表明少量样本中 support routing 改动仍会损伤 answer 表述线索。

## 7. Oracle Gap

oracle diagnostic：

```text
queries_with_actions = 996
oracle_best_joint_delta = +0.1411
oracle_best_answer_safe_joint_delta = +0.1508
oracle_support_delta = +0.0954
positive_candidate_rate = 0.2239
answer_safe_positive_candidate_rate = 0.2239
selector_recall_of_positive_candidates = 0.1839
```

含义：

候选池仍然有很大的论文级潜力。v2.2 的问题不是没有可用 candidate，而是 selector 对 positive candidate 的召回太低。当前 selector 只能召回约 18.4% 的 positive candidates。

## 8. 当前结论

v2.2 是一个实质性进展，但不是最终主结果。

正向进展：

1. final_1000 不再退化成 baseline；
2. fallback 从 v2.1 的 0.92 降到 0.50；
3. selected action 全部有效；
4. support_recall 与 sp_f1 出现明确正信号；
5. joint_f1 有 +0.0081 的正趋势。

不能宣称成功的原因：

1. strict gate 仍为 false；
2. answer_f1_delta 为 `-0.000072`，虽极小但为负；
3. joint_f1 显著性不足；
4. positive candidate recall 仍低。

## 9. 下一步建议

建议进入 `selector_v2.3_answer_neutral_positive_selector`：

1. 用 v2.2 的 effective action table 训练 answer-neutral positive action selector；
2. 目标从 `support_first` 改成 `joint_gain_first with answer_nonnegative_constraint`；
3. 显式惩罚 answer_f1 下降 action；
4. 提高 positive candidate recall，目标从 `0.1839` 提升到 `0.30-0.40`；
5. 继续使用 query-level cross-fit，保持 strict no-leak。

一句话判断：v2.2 已经把“规模校准失败”修到有可见正信号，但最终论文主结果还需要 v2.3 解决 answer-neutral 选择与 positive candidate recall。
