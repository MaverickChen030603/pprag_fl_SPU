# V7-HP-PAPER selector_v2.3 当前进展报告

查询时间：2026-06-19 19:16 JST  
实验目录：`/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/selector_v2_3/`

## 1. 当前状态

服务器 live check 显示：

```text
selector_v2_3 相关进程：无
pipeline 状态：已完成
hard error：未发现
报告镜像：已存在
```

当前 v2.3 的 labels、model_cv、calibration、final_1000、significance、ablation、diagnostics 与 paper report 均已生成。`实验分析报告/V7-HP-PAPER/` 下已有 latest 镜像。

## 2. 数据与标签分布

v2.3 复用 v2.2 的 effective action table，不重跑 reader。

```text
num_actions = 5000
num_queries = 1000
answer_safe_rate = 0.9468
joint_positive_rate = 0.0914
answer_safe_joint_positive_rate = 0.0908
paper_positive_rate = 0.0896
answer_drop_rate = 0.0532
queries_with_no_positive_action = 778
```

解释：候选池中 paper-positive action 很稀疏，只有约 8.96% 的 action 是 paper-positive；1000 个 query 中有 778 个没有 paper-positive action。因此 v2.3 的任务本质上是从很稀疏的正例中提升 recall，同时避免 answer_f1 下降。

## 3. Final 1000 Cross-Fit 结果

v2.3 final_1000 结果：

| metric | baseline | selector_v2.3 | delta |
|---|---:|---:|---:|
| answer_access@k | 0.8330 | 0.8450 | +0.0120 |
| support_recall@k | 0.8190 | 0.8380 | +0.0190 |
| sp_f1 | 0.7483 | 0.7737 | +0.0254 |
| answer_em | 0.4800 | 0.4810 | +0.0010 |
| answer_f1 | 0.6100 | 0.6122 | +0.0023 |
| joint_f1 | 0.5170 | 0.5320 | +0.0150 |

选择行为：

```text
selected_count = 500 / 1000
fallback_rate = 0.5000
effective_selected_count = 500
selected_effective_action_rate = 1.0000
positive_candidate_recall = 0.3288
answer_safe_positive_candidate_recall = 0.3274
selected_answer_drop_rate = 0.0580
selected_joint_positive_rate = 0.1460
gate_pass = true
paper_main_recommended = true
```

fold 配置分布：

```text
two_stage + all_effective_conservative + selected_fraction=0.5: 2 folds
pairwise_ranker + all_effective_conservative + selected_fraction=0.5: 3 folds
```

## 4. 显著性

paired bootstrap：

| metric | mean_delta | 95% CI | p-value |
|---|---:|---:|---:|
| answer_f1 | +0.0023 | [-0.0114, +0.0158] | 0.3625 |
| joint_f1 | +0.0150 | [+0.0001, +0.0302] | 0.0245 |
| support_recall@5 | +0.0190 | [+0.0085, +0.0295] | 0.0000 |
| sp_f1 | +0.0254 | [+0.0106, +0.0393] | 0.0000 |

解释：v2.3 的 `joint_f1`、`support_recall@5`、`sp_f1` 均达到统计正信号；`answer_f1` 为正但不显著。论文表述应强调 answer-neutral 保护成功，但不要声称 answer_f1 显著提升。

## 5. 与 v2.2 的关键差异

v2.2：

```text
answer_f1_delta = -0.0001
joint_f1_delta = +0.0081
support_recall_delta = +0.0075
sp_f1_delta = +0.0103
positive_candidate_recall = 0.1839
gate_pass = false
```

v2.3：

```text
answer_f1_delta = +0.0023
joint_f1_delta = +0.0150
support_recall_delta = +0.0190
sp_f1_delta = +0.0254
positive_candidate_recall = 0.3288
gate_pass = true
```

核心突破：v2.3 把 positive candidate recall 从 0.1839 提升到 0.3288，同时把 answer_f1 从微负转为正，并使 joint_f1 达到显著正增益。

## 6. Ablation 摘要

主要 ablation：

```text
two_stage:
  answer_f1_delta = +0.0051
  joint_f1_delta = +0.0113
  support_recall_delta = +0.0120
  sp_f1_delta = +0.0150
  positive_candidate_recall = 0.2928
  gate_pass = true

paper_positive_classifier:
  answer_f1_delta = +0.0017
  joint_f1_delta = +0.0076
  support_recall_delta = +0.0120
  sp_f1_delta = +0.0150
  positive_candidate_recall = 0.2838
  gate_pass = true

answer_drop_rejector_support_ranker:
  answer_f1_delta = -0.0005
  joint_f1_delta = +0.0078
  selected_effective_action_rate = 0.646
  positive_candidate_recall = 0.1216
  gate_pass = false

constrained_regression:
  answer_f1_delta = +0.0050
  joint_f1_delta = +0.0092
  selected_effective_action_rate = 0.962
  positive_candidate_recall = 0.2883
  gate_pass = false
```

当前最稳主结果是 cross-fit 自动选择的 `two_stage / pairwise_ranker` 混合配置；单独 two-stage 也通过 gate，但 joint_f1 不如主配置。

## 7. Failure Diagnosis

失败统计：

```text
candidate_pool_no_positive_action = 778
selected_positive = 73
positive_action_available_but_not_selected = 102
wrong_action_selected = 41
answer_drop_selected = 2
support_positive_but_joint_negative = 4
```

解释：

1. 最大限制仍是候选池本身：778 个 query 没有 paper-positive action；
2. 在有 positive action 的 query 中，v2.3 已召回 0.3288，比 v2.2 明显提升；
3. 仍有 102 个 positive action available but not selected，是后续可优化空间；
4. answer_drop_selected 只有 2 类 case，说明 answer-neutral 约束总体有效。

## 8. 当前结论

v2.3 已完成并通过 gate，可作为论文主结果候选。

建议论文主表述：

```text
Under strict no-leak query-level cross-fitting, the answer-neutral positive selector converts the support-side routing signal into statistically significant joint_f1 and support-side gains while preserving answer_f1.
```

需要谨慎的边界：

1. answer_f1 是正向但不显著；
2. selector 成功依赖已有 candidate pool，仍有大量 query 没有 paper-positive action；
3. 后续若补实验，优先做 candidate_pool_quality_breakdown 与 positive_candidate_feature_importance，而不是再盲目大规模 reader validation。

## 9. 当前报告路径

正式报告：

```text
V7-HP-PAPER/selector_v2_3/reports/v7_hp_paper_selector_v2_3_report.md
```

当前进展报告：

```text
V7-HP-PAPER/selector_v2_3/reports/v7_hp_paper_selector_v2_3_current_report_20260619_1916.md
```

镜像目录：

```text
实验分析报告/V7-HP-PAPER/
```
