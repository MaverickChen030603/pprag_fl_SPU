# V7-HP-PAPER / 2Wiki 当前进展报告

生成时间：2026-06-29 13:53 JST  
项目根目录：`/home/iiserver31/projects/FedE4RAG-main`  
当前关注范围：`V7-HP-PAPER/cross_dataset_validation`

## 1. 实时运行状态

服务器实时检查结果：

- 当前没有 `2wiki`、`V7-HP-PAPER`、`positive_action_detectability`、`bm25_anchor`、`selector_alignment`、`run_2wiki` 相关进程在运行。
- 最新 2Wiki 相关产物停在 2026-06-24 14:38 JST 完成的 `2wiki_positive_action_detectability_audit`。
- 没有启动 2Wiki 1000 reader validation。
- 没有启动 MuSiQue。
- HotpotQA v2.3 主结果未被修改。

GPU 当前均有外部占用：

| GPU | 型号 | 显存占用 | 利用率 |
|---|---|---:|---:|
| 0 | NVIDIA A100-PCIE-40GB | 33938 / 40960 MB | 30% |
| 1 | NVIDIA A100-PCIE-40GB | 37690 / 40960 MB | 37% |
| 2 | NVIDIA A100-PCIE-40GB | 37690 / 40960 MB | 41% |
| 3 | NVIDIA A100-PCIE-40GB | 38685 / 40960 MB | 35% |

当前判断：2Wiki 外部验证线处于“已完成诊断、等待论文整合”的状态，而不是运行中状态。

## 2. 最近完成的实验分支

最新完成分支：

`V7-HP-PAPER/cross_dataset_validation/2wiki_positive_action_detectability_audit/`

该分支目标不是继续调参，也不是扩大评估，而是回答一个论文落地问题：

> 为什么 2Wiki 上存在 oracle positive actions，但当前 no-leak selector 不能稳定识别这些超过 BM25 strong baseline 的动作？

本轮已完成：

- 汇总已有 2Wiki alignment / BM25-anchor repair 结果；
- 构建 action-outcome 表；
- 分析 positive action feature margin；
- 分析 candidate pool vs BM25；
- 分析 selector recall failure；
- 分析 safety predictor weakness；
- 导出 case studies；
- 导出论文表格；
- 撰写 limitation / appendix draft。

## 3. 关键结果摘要

### 3.1 汇总指标

| 指标 | 数值 |
|---|---:|
| dev queries | 300 |
| action rows | 1800 |
| oracle positive-vs-BM25 queries | 73 / 300 |
| oracle positive rate | 0.2433 |
| candidate_pool_no_positive_vs_BM25 | 227 / 300 |
| oracle best answer delta vs BM25 | +0.1615 |
| oracle best evidence delta vs BM25 | +0.1267 |
| oracle best joint delta vs BM25 | +0.1533 |
| best no-leak selector | `bm25_anchor_answer_neutral_selector` |
| best no-leak answer delta vs BM25 | +0.0017 |
| best no-leak evidence F1 delta vs BM25 | +0.0073 |
| best no-leak joint delta vs BM25 | +0.0002 |
| selected effective action rate | 0.7000 |
| original repair positive recall | 0.0433 |
| safety predictor answer-safe AUC | 0.5567 |
| safety predictor paper-positive AUC | 0.5451 |

### 3.2 两个 positive 口径必须区分

本轮报告明确区分两个口径：

1. Oracle query-level positive：`73 / 300`
   - 来自 oracle gap diagnostic；
   - 表示这些 query 在更宽候选空间中存在超过 BM25 的动作；
   - 只能作为 diagnostic，不是 inference-time 方法。

2. Strict action-table positive：`33 / 300`
   - 来自 BM25-anchor action table 中实际暴露给 no-leak selector 的动作；
   - 这是 feature margin / selector recall 分析的可观测 action-level 口径；
   - 说明许多 oracle opportunity 并没有进入当前 selector 可识别的动作空间。

这个口径差异是当前最重要的诊断发现之一：问题不只是 selector 不会选，也包括 candidate/action generation 没把足够多的 positive opportunity 暴露给 selector。

## 4. Candidate Pool 诊断

Candidate pool 分析结果：

| 指标 | 数值 |
|---|---:|
| oracle queries with positive vs BM25 | 73 |
| oracle queries without positive vs BM25 | 227 |
| strict action-label queries with positive | 33 |
| strict action-label queries without positive | 267 |
| strict action-label positive rate | 0.1100 |
| BM25 already strong cases | 81 |
| evidence gain without joint gain cases | 13 |
| answer gain without evidence gain cases | 15 |

按 candidate family 的 strict positive rate：

| family | positive rate |
|---|---:|
| bridge_insert | 0.0667 |
| replace_slot4_or_5 | 0.0667 |
| replace_slot5_only | 0.0633 |
| tail_swap_evidence | 0.0633 |
| bm25_fallback | 0.0000 |
| bm25_no_change_control | 0.0000 |

结论：

> 2Wiki 的主要瓶颈不是 BM25-anchor 之后仍然严重破坏 answer anchor，而是当前 candidate pool 中大多数 query 根本没有可超过强 BM25 baseline 的动作。

所以，不建议继续扩大到 1000；下一步若做研究扩展，应优先改 candidate generation，而不是继续调 selector 阈值。

## 5. Positive Action Feature Margin

Feature margin 总判断：

> positive actions are weakly distinguishable with current features.

Top univariate effects：

| feature | effect size | AUC | rho with joint |
|---|---:|---:|---:|
| num_added_docs | 1.0594 | 0.6797 | -0.0381 |
| answer_risk_score | 1.0342 | 0.6742 | -0.0406 |
| num_removed_docs | 1.0067 | 0.6759 | -0.0456 |
| bm25_score_delta | -0.3624 | 0.2802 | 0.0403 |
| support_proxy_delta_vs_bm25 | -0.2995 | 0.3377 | 0.0169 |
| evidence_proxy_delta_vs_bm25 | -0.2769 | 0.3421 | 0.0078 |

解释：

- 一些结构特征能分出“发生了替换/插入”的动作，但这些特征与 joint utility 的相关性并不稳定。
- support/evidence proxy 与 reader joint utility 在 2Wiki 上没有形成足够强的一致信号。
- 这支持“跨数据集 selector feature 不可直接泛化”的 limitation 写法。

## 6. Selector Recall Failure

在 strict action-table positive 口径下：

| 指标 | 数值 |
|---|---:|
| oracle positive queries | 73 |
| strict action-label positive queries | 33 |
| selected positive queries | 13 |
| positive recall over strict positives | 0.3939 |
| positive available but not selected | 20 |
| oracle-positive but no strict positive action in anchor table | 49 |
| fallback on positive query | 14 |
| wrong action on positive query | 6 |
| answer-drop selected | 0 |

Best-positive predicted rank distribution：

| rank | count |
|---|---:|
| 1 | 3 |
| 3 | 19 |
| 4 | 2 |
| 5 | 2 |
| 6 | 7 |

判断：

- 一部分 positive action 已能被 selector 找到，但 recall 仍不足。
- 很多 best positive 排在第 3 或更后，说明 ranker 分数没有把真实有益动作推到最前。
- `answer_drop_selected = 0` 说明 BM25-anchor 已经基本解决 answer-anchor 破坏问题；剩余问题主要是 candidate exposure 和 positive ranking。

## 7. Safety Predictor 弱点

Safety predictor 结果：

| 指标 | 数值 |
|---|---:|
| answer_safe_auc | 0.5567 |
| paper_positive_auc | 0.5451 |
| answer_safe_rate | 0.9633 |
| paper_positive_rate | 0.0433 |
| false_safe_cases | 66 |
| false_negative_cases | 0 |
| positive safe_answer_prob mean | 0.9365 |
| non-positive safe_answer_prob mean | 0.9470 |
| answer-drop safe_answer_prob mean | 0.9429 |

判断：

- AUC 接近 0.55，不能声称 safety predictor 在 2Wiki 上可靠。
- positive 和 non-positive 的 `safe_answer_prob` 均值几乎没有可用分离。
- answer-drop actions 的 safe probability 也很高，说明当前 answer-neutral calibration 对 2Wiki 不够敏感。

论文写法应为：

> Cross-dataset answer-neutral calibration is dataset-sensitive.

## 8. 当前论文建议

当前建议不变：

1. 冻结 HotpotQA v2.3 作为论文主结果。
2. 2Wiki 不作为 selector-level generalization success。
3. 2Wiki 可作为：
   - external sanity check；
   - dataset adapter / reader-backed evaluation pipeline validation；
   - lexical/BM25 routing strong baseline evidence；
   - cross-dataset selector limitation；
   - appendix diagnostic。
4. 不建议启动 2Wiki 1000 validation，因为 smoke/diagnostic 结果已经说明主瓶颈在 candidate/action exposure 和 feature detectability，而不是样本量不足。
5. 不建议启动 MuSiQue，至少在 2Wiki 的 selector/BM25 问题解释清楚前不扩展新数据集。

## 9. 可直接写入论文的安全表述

英文 limitation draft 已生成，核心表述如下：

> We further tested the pipeline on 2WikiMultiHopQA as an external sanity check. A strong lexical/BM25 baseline substantially improved reader-backed evidence and joint metrics over the raw context order, indicating that the dataset adapter and reader evaluation pipeline transfer correctly. However, when evaluated against this strong BM25 baseline, the HotpotQA-trained selector and the 2Wiki cross-fitted selector did not establish reliable selector-level generalization. A BM25-anchor repair reduced negative transfer and nearly matched BM25, but the gain was too small to justify a full 1000-sample validation. Oracle diagnostics show that positive actions beyond BM25 exist, but the current no-leak features and safety predictor do not identify them reliably. We therefore report 2Wiki as a diagnostic limitation rather than as a main generalization claim.

## 10. 当前产物路径

Detectability audit 主目录：

- `V7-HP-PAPER/cross_dataset_validation/2wiki_positive_action_detectability_audit/`

关键报告：

- `reports/2wiki_positive_action_detectability_report.md`
- `reports/2wiki_paper_limitation_section_draft.md`
- `reports/2wiki_appendix_diagnostic.md`

关键输出：

- `outputs/collected/2wiki_collected_summary.json`
- `outputs/feature_margin/feature_margin_summary.json`
- `outputs/candidate_pool/candidate_pool_vs_bm25_summary.json`
- `outputs/selector_recall/selector_recall_failure_summary.json`
- `outputs/safety_predictor/safety_predictor_weakness_summary.json`
- `outputs/case_studies/case_studies.json`
- `outputs/tables/2wiki_main_status_table.md`
- `outputs/tables/2wiki_oracle_gap_table.md`
- `outputs/tables/2wiki_selector_repair_table.md`
- `outputs/tables/2wiki_failure_summary_table.md`
- `outputs/tables/2wiki_feature_margin_table.md`

中文归档镜像：

- `实验分析报告/V7-HP-PAPER/2wiki_positive_action_detectability_report_latest.md`
- `实验分析报告/V7-HP-PAPER/2wiki_paper_limitation_section_draft_latest.md`
- `实验分析报告/V7-HP-PAPER/2wiki_appendix_diagnostic_latest.md`

本次当前报告建议写入：

- `V7-HP-PAPER/cross_dataset_validation/2wiki_positive_action_detectability_audit/reports/2wiki_current_progress_report_20260629.md`
- `V7-HP-PAPER/cross_dataset_validation/2wiki_positive_action_detectability_audit/reports/2wiki_current_progress_report_latest.md`
- `实验分析报告/V7-HP-PAPER/2wiki_current_progress_report_latest.md`

## 11. 总结

截至 2026-06-29，2Wiki 分支已经从“尝试外部泛化”转为“完成外部 limitation 诊断”。最关键的论文价值不是宣称 2Wiki 成功，而是解释为什么跨数据集 selector 难以超过强 BM25：

- BM25 本身很强；
- candidate pool 大多数 query 没有正动作；
- oracle 机会未充分暴露给 no-leak selector；
- 当前特征对 positive action 的区分力弱；
- safety predictor 跨数据集校准弱；
- BM25-anchor 能避免负迁移，但只能把结果修到接近 BM25。

因此，当前最稳妥的论文路线是：HotpotQA v2.3 做主结果，2Wiki 做 external diagnostic / limitation / future work。
