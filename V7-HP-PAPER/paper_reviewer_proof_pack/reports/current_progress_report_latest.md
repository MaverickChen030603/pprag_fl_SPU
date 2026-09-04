# V7-HP-PAPER 当前进展报告

生成时间：2026-06-29 14:38 JST  
项目根目录：`/home/iiserver31/projects/FedE4RAG-main`  
当前阶段：论文实验收束与审稿防守包整理

## 1. 实时运行状态

服务器实时检查结果：

- 当前没有 `V7-HP-PAPER`、`reviewer_proof`、`paper_integration`、`2wiki`、`selector_v2_3`、`run_2wiki`、`MuSiQue` 相关进程在运行。
- 没有启动新的 reader validation。
- 没有启动 2Wiki 1000。
- 没有启动 MuSiQue。
- HotpotQA v2.3 主结果未被修改。

GPU 当前状态：

| GPU | 型号 | 显存占用 | 利用率 |
|---|---|---:|---:|
| 0 | NVIDIA A100-PCIE-40GB | 36609 / 40960 MB | 15% |
| 1 | NVIDIA A100-PCIE-40GB | 37690 / 40960 MB | 27% |
| 2 | NVIDIA A100-PCIE-40GB | 37690 / 40960 MB | 32% |
| 3 | NVIDIA A100-PCIE-40GB | 38685 / 40960 MB | 22% |

判断：V7-HP-PAPER 当前不是运行中实验，而是已完成结果冻结、论文整合和 reviewer proof pack 的收束状态。

## 2. 最新完成工作

最新完成分支：

`V7-HP-PAPER/paper_reviewer_proof_pack/`

完成时间：2026-06-29 14:20 JST

本分支目标是补齐审稿防守材料，而不是追求新主结果。已完成：

- 输入完整性检查；
- HotpotQA selector stability 分析；
- threshold / selected_fraction sensitivity 分析；
- fair baseline comparison；
- HotpotQA success / answer-preserving / failure case studies；
- 2Wiki limitation case studies；
- reviewer risk table；
- experiment sufficiency memo；
- reviewer response prep；
- final additional experiment recommendation。

## 3. 当前主结果状态

HotpotQA v2.3 仍冻结为论文主结果：

| 指标 | 数值 |
|---|---:|
| answer_f1_delta | +0.0023 |
| joint_f1_delta | +0.0150 |
| support_recall@5_delta | +0.0190 |
| sp_f1_delta | +0.0254 |
| fallback_rate | 0.5000 |
| positive_candidate_recall | 0.3288 |
| gate_pass | true |
| paper_main_recommended | true |

显著性边界保持不变：

- `joint_f1`、`support_recall@5`、`sp_f1` 可以写显著提升；
- `answer_f1` 只能写 answer-preserving / small non-significant positive delta；
- 不能写 answer_f1 significantly improves。

## 4. Stability 结论

`paper_reviewer_proof_pack` 已输出 fold-level stability：

| fold | model | answer_delta | joint_delta | support_delta | sp_delta | gate |
|---|---|---:|---:|---:|---:|---|
| 0 | two_stage | -0.0140 | -0.0127 | +0.0125 | +0.0157 | false |
| 1 | pairwise_ranker | +0.0197 | +0.0507 | +0.0275 | +0.0371 | true |
| 2 | two_stage | +0.0067 | +0.0080 | +0.0100 | +0.0143 | true |
| 3 | pairwise_ranker | -0.0030 | +0.0051 | +0.0050 | +0.0071 | false |
| 4 | pairwise_ranker | +0.0019 | +0.0239 | +0.0400 | +0.0529 | true |

解释：

- 4 / 5 folds 的 `joint_f1_delta` 为正；
- 5 / 5 folds 的 support-side 指标为正；
- fold 0 是主要 caveat：support gain 没能转成 joint gain，且 answer_f1 下降；
- 因此审稿表述应为“整体效果不是单一 fold 驱动，但存在 fold-level variability”。

## 5. Threshold / Sensitivity 结论

已从 calibration / model_cv 现有记录整理 threshold sensitivity，不重跑 reader。

关键发现：

- `selected_fraction=0.5` 在 calibration top configs 中反复出现；
- `answer_safe_threshold=0.5` 与 `positive_threshold=0.1` 是多 fold 中稳定出现的 answer-neutral 配置；
- 附近配置也能产生正向 joint/support 信号；
- 因此 v2.3 不是单一偶然阈值点，但 answer-neutral calibration 仍应作为 paper method choice 而非普适最优定律来写。

## 6. Fair Baseline 结论

已输出 `fair_baseline_comparison_table.md`。

核心对比：

| 方法 | answer_delta | joint_delta | support_delta | sp_delta | gate | paper_role |
|---|---:|---:|---:|---:|---|---|
| baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | reference | reference_baseline |
| v2.2 support-first | -0.0001 | +0.0081 | +0.0075 | +0.0103 | false | failed_variant |
| v2.3 answer-neutral positive selector | +0.0023 | +0.0150 | +0.0190 | +0.0254 | true | main_result |
| two_stage | +0.0051 | +0.0113 | +0.0120 | +0.0150 | true | ablation |
| paper_positive_classifier | +0.0017 | +0.0076 | +0.0120 | +0.0150 | true | ablation |
| answer_drop_rejector_support_ranker | -0.0005 | +0.0078 | +0.0125 | +0.0166 | false | failed_variant |
| constrained_regression | +0.0050 | +0.0092 | +0.0105 | +0.0150 | false | failed_variant |
| no_safety_predictor | -0.0070 | +0.0040 | +0.0110 | +0.0136 | false | failed_variant |
| no_support_features | +0.0002 | +0.0127 | +0.0135 | +0.0184 | true | ablation |
| random_effective_action | NA | NA | NA | NA | not_available_without_new_reader | not_available |

判断：

- v2.3 相比 support-first 有更强 joint/support 转化；
- no_safety_predictor 说明 answer-neutral / safety 约束必要；
- no_support_features 仍有正信号，但弱于 v2.3，说明 support/routing features 有贡献；
- random_effective_action 无现成 reader 结果，已按要求标为 `not_available_without_new_reader`，没有补跑。

## 7. 2Wiki 状态

2Wiki 仍保持 external diagnostic / limitation / appendix 定位。

固定结论：

- adapter 和 reader pipeline 成功；
- BM25 lexical routing 明显优于 raw context；
- selector-level generalization beyond BM25 未建立；
- BM25-anchor repair 只接近 BM25，增益太小；
- positive action detectability audit 说明瓶颈在 candidate exposure、feature separability 和 safety calibration；
- 不建议启动 2Wiki 1000；
- 不建议启动 MuSiQue。

论文边界：

> 2Wiki verifies the adapter and reader-backed evaluation pipeline, but it should be reported as a diagnostic limitation rather than as a main generalization claim.

## 8. Reviewer Proof Pack 已完成产物

主目录：

`V7-HP-PAPER/paper_reviewer_proof_pack/`

关键报告：

- `reports/experiment_sufficiency_memo.md`
- `reports/final_additional_experiment_recommendation.md`
- `reports/reviewer_risk_memo.md`
- `reports/reviewer_response_prep.md`

关键表格：

- `outputs/tables/hotpot_stability_table.md`
- `outputs/tables/threshold_sensitivity_table.md`
- `outputs/tables/fair_baseline_comparison_table.md`
- `outputs/tables/reviewer_risk_table.md`

关键案例：

- `outputs/case_studies/hotpot_success_cases.md`
- `outputs/case_studies/hotpot_answer_preserving_cases.md`
- `outputs/case_studies/hotpot_failure_cases.md`
- `outputs/case_studies/2wiki_limitation_cases.md`

中文归档镜像：

`实验分析报告/V7-HP-PAPER/paper_reviewer_proof_pack/`

## 9. 当前实验充分性判断

当前 proof pack 的明确结论：

> Current experiments are sufficient for a HotpotQA-centered paper with 2Wiki diagnostic limitation.

同时必须保留边界：

> Current experiments are not sufficient for a strong cross-dataset generalization claim.

因此下一步不建议做新大规模实验。当前最合适的动作是进入论文写作、图表压缩、主文/appendix 分配和 reviewer response 预案整理。

## 10. 下一步建议

优先级建议：

1. 将 HotpotQA main result table、significance table、fair baseline comparison 放入主文。
2. 将 stability、threshold sensitivity、case studies 放入 appendix。
3. 将 2Wiki diagnostic limitation 放入 limitation 或 appendix。
4. 保持 claim boundary memo 的表述，不宣称 answer_f1 显著提升，也不宣称 2Wiki selector 泛化成功。
5. 不启动 v2.4、2Wiki 1000 或 MuSiQue。

## 11. 本次报告路径

建议写入：

- `V7-HP-PAPER/paper_reviewer_proof_pack/reports/current_progress_report_20260629_1438.md`
- `V7-HP-PAPER/paper_reviewer_proof_pack/reports/current_progress_report_latest.md`
- `实验分析报告/V7-HP-PAPER/current_progress_report_latest.md`
- `实验分析报告/V7-HP-PAPER/paper_reviewer_proof_pack/current_progress_report_latest.md`
