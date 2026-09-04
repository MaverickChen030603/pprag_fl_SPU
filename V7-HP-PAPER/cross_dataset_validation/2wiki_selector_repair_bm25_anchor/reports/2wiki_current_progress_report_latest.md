# V7-HP-PAPER / 2Wiki 当前进展报告

生成时间：2026-06-24 14:28 JST  
项目根目录：`/home/iiserver31/projects/FedE4RAG-main`  
当前分支范围：`V7-HP-PAPER/cross_dataset_validation`

## 1. 实时运行状态

服务器实时检查结果显示：当前没有 `2wiki`、`V7-HP-PAPER`、`bm25_anchor`、`selector_alignment`、`run_2wiki` 相关进程在运行。

GPU 当前均有占用：

| GPU | 型号 | 显存占用 | 利用率 |
|---|---|---:|---:|
| 0 | NVIDIA A100-PCIE-40GB | 34062 / 40960 MB | 39% |
| 1 | NVIDIA A100-PCIE-40GB | 37792 / 40960 MB | 43% |
| 2 | NVIDIA A100-PCIE-40GB | 37792 / 40960 MB | 34% |
| 3 | NVIDIA A100-PCIE-40GB | 38625 / 40960 MB | 41% |

结论：2Wiki 外部验证相关任务已经完成并停在 smoke / repair 分析阶段；当前没有新的 1000-sample reader job 在跑。

## 2. 已完成工作线

### 2.1 2Wiki 数据准备

2WikiMultiHopQA dev/test 已经完成服务器侧准备，包含：

- `answer`
- `context docs`
- `supporting_facts`
- `supporting_titles`
- `evidences`

关键产物：

- `outputs/2wiki_adapter/2wiki_dev_converted.json`
- `outputs/2wiki_adapter/2wiki_test_converted.json`
- `outputs/2wiki_adapter/adapter_summary.json`

数据层结论：dev split 可用于 answer / support / joint 指标；test split 不适合作为 support/joint 主评估，因为缺少完整句级 support/evidence 标签。

### 2.2 2Wiki dev-300 非 reader smoke

已完成 dev-300 retrieval/access smoke。

| 方法 | answer_access@5 | support_recall@5 | all_support_access@5 | joint_access@5 |
|---|---:|---:|---:|---:|
| context order | 0.5433 | 0.4775 | 0.1667 | 0.1533 |
| lexical BM25 top5 | 0.7133 | 0.7967 | 0.5467 | 0.5067 |
| BM25 - context | +0.1700 | +0.3192 | +0.3800 | +0.3533 |

结论：2Wiki adapter 和 basic lexical routing 管线是有效的，BM25 相对原始 context order 有明显优势。

### 2.3 2Wiki dev-300 reader-backed smoke

已接入 frozen selector/reader pipeline，使用 `google/flan-t5-large` reader，完成 dev-300 reader-backed smoke。

| 方法 | answer_access@5 | support_recall@5 | sp_f1 | answer_em | answer_f1 | joint_f1 |
|---|---:|---:|---:|---:|---:|---:|
| context order | 0.5867 | 0.4775 | 0.3761 | 0.3133 | 0.3714 | 0.1669 |
| frozen_selector_bm25_top5 | 0.7633 | 0.7967 | 0.7290 | 0.4300 | 0.4977 | 0.4176 |
| delta | +0.1766 | +0.3192 | +0.3529 | +0.1167 | +0.1263 | +0.2507 |

结论：reader 管线可用；BM25/lexical routing 能带来真实 QA 增益。但这不是 Hotpot v2.3 selector 的跨数据集泛化证据，只能说明 2Wiki adapter + reader + lexical retrieval 工作正常。

## 3. Selector Alignment 分支结果

分支目录：

`V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment`

已完成：

- action table 300
- v2.3-style feature build
- selector smoke 300
- ablation
- no-leak audit
- failure diagnosis

核心结果：

| 方法 | answer_f1 | evidence_f1 | joint_f1 | joint delta vs BM25 |
|---|---:|---:|---:|---:|
| BM25 baseline | 0.4469 | 0.7270 | 0.3790 | 0.0000 |
| Hotpot v2.3 frozen transfer | 0.4300 | 0.6614 | 0.3449 | -0.0341 |
| 2Wiki v2.3 crossfit selector | 0.3660 | 0.4003 | 0.1855 | -0.1935 |
| oracle diagnostic only | 0.6055 | 0.7898 | 0.5323 | +0.1533 |

Gate 结果：`stop_at_smoke_300`。

诊断：

- `positive_action_available_but_not_selected`: 142
- `selector_underperforms_bm25`: 55
- `answer_drop_selected`: 59
- `support_positive_but_joint_negative`: 12

结论：原始 selector alignment 没有超过强 BM25 baseline。Oracle 说明候选动作里存在可提升空间，但当前 no-leak selector 找不到这些动作。

## 4. BM25-anchor repair 分支结果

分支目录：

`V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor`

### 4.1 Oracle gap audit

| 指标 | 数值 |
|---|---:|
| queries | 300 |
| positive vs BM25 queries | 73 |
| positive rate | 0.2433 |
| oracle best answer delta vs BM25 | +0.1615 |
| oracle best evidence delta vs BM25 | +0.1267 |
| oracle best joint delta vs BM25 | +0.1533 |
| selector recall of positive vs BM25 | 0.3425 |

解释：候选动作空间确实存在比 BM25 更好的动作，约 24.33% query 有提升机会；但这是 oracle diagnostic，不是推理时可用方法。

### 4.2 BM25-anchor action table

| 指标 | 数值 |
|---|---:|
| queries | 300 |
| actions | 1800 |
| actions/query | 6.0 |
| effective_action_rate | 0.6589 |
| BM25 top1 preserve rate | 1.0000 |
| BM25 top2 preserve rate | 1.0000 |
| BM25 top3 preserve rate | 1.0000 |
| hard rule violations | 0 |

解释：BM25-anchor 设计成功保住 BM25 前 1-3 个 reader anchor，解决了此前 selector 破坏答案锚点的问题。

### 4.3 Safety predictor

| 指标 | 数值 |
|---|---:|
| answer_safe_auc | 0.5567 |
| paper_positive_auc | 0.5451 |
| false_safe_rate | 0.0367 |
| false_negative_rate | 0.0000 |
| answer_safe_rate | 0.9633 |
| paper_positive_rate | 0.0433 |

解释：safety predictor 有一点信号，但 AUC 偏弱，只能视为 smoke-level calibration。

### 4.4 Reader smoke 300 主结果

| 方法 | answer_f1 | evidence_recall@5 | evidence_f1 | joint_f1 | joint delta vs BM25 | effective rate | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 / lexical routing | 0.4469 | 0.7933 | 0.7270 | 0.3790 | 0.0000 | 0.0000 | 1.0000 |
| previous v23 crossfit selector | 0.3660 | 0.4992 | 0.4003 | 0.1855 | -0.1935 | 0.2633 | 0.0000 |
| bm25_anchor_support_first | 0.4430 | 0.7875 | 0.7203 | 0.3751 | -0.0039 | 0.1333 | 0.8667 |
| bm25_anchor_safety_first | 0.4325 | 0.7858 | 0.7165 | 0.3599 | -0.0191 | 1.0000 | 0.0000 |
| bm25_anchor_positive_selector | 0.4389 | 0.7850 | 0.7146 | 0.3650 | -0.0140 | 1.0000 | 0.0000 |
| bm25_anchor_answer_neutral_selector | 0.4486 | 0.7983 | 0.7343 | 0.3792 | +0.0002 | 0.7000 | 0.3000 |
| no_safety_predictor | 0.4293 | 0.7950 | 0.7295 | 0.3600 | -0.0190 | 1.0000 | 0.0000 |
| no_support_features | 0.4339 | 0.7783 | 0.7038 | 0.3570 | -0.0220 | 1.0000 | 0.0000 |
| oracle diagnostic only | 0.5063 | 0.8167 | 0.7559 | 0.4372 | +0.0582 | 0.1167 | 0.8833 |

最佳 no-leak repair variant 是 `bm25_anchor_answer_neutral_selector`：

- answer_f1: +0.0017 vs BM25
- evidence_recall@5: +0.0050 vs BM25
- evidence_f1: +0.0073 vs BM25
- joint_f1: +0.0002 vs BM25

但它没有通过扩展 gate，因为：

- selected_effective_action_rate = 0.7000，低于可靠扩展所需水平；
- positive_vs_bm25_recall = 0.0433，说明真正有益动作仍多数没有被 selector 捕获；
- joint_f1 增益只有 +0.0002，无法作为 1000-sample 正式泛化依据。

### 4.5 Failure diagnosis

| failure type | count |
|---|---:|
| candidate_pool_no_positive_vs_bm25 | 227 |
| positive_vs_bm25_available_but_not_selected | 27 |
| answer_drop_selected | 1 |
| total failures | 255 |

解释：

1. 最大瓶颈不是 reader 被坏文档严重干扰，而是候选池里大多数 query 根本没有比 BM25 更好的可选动作。
2. 对于有 positive action 的 query，selector 仍经常选不中。
3. BM25-anchor 已经大幅减少 answer anchor 破坏，`answer_drop_selected` 只剩 1 个，但同时也让 selector 更保守。

### 4.6 No-leak audit

No-leak audit 状态：`passed`。

已确认：

- query fold disjoint；
- safety predictor 仅使用 train fold；
- threshold calibration 没有启动正式 1000；
- held-out outcome 未用于 inference；
- gold answer/support 未作为 inference feature；
- oracle 与正式方法分离。

## 5. 当前判断

当前 2Wiki 外部验证应分三层写入论文或报告：

1. 可以写：2Wiki 数据 adapter、reader-backed smoke、BM25 lexical routing 通过，说明跨数据集评估管线和 reader 评估链路可用。
2. 可以写为 limitation：Hotpot v2.3 selector 不能直接泛化到 2Wiki strong BM25 baseline；原始 selector 和 crossfit selector 都显著弱于 BM25。
3. 可以写为诊断发现：BM25-anchor repair 证明“保留答案锚点”是必要的，能把 selector 从明显负迁移修复到接近 BM25，并带来极小正信号，但还不足以作为正式 selector-level generalization claim。

因此，当前 paper-safe 结论是：

> 2Wiki candidate actions contain positive opportunities beyond BM25, but the current no-leak selector does not identify them reliably enough. The result should be reported as pipeline / lexical-routing validation plus cross-dataset selector limitation, not as main selector generalization evidence.

## 6. 是否启动 1000 validation

不建议启动 2Wiki 1000 validation。

原因：

- smoke gate 仍是 `stop_at_smoke_300`；
- 最佳 no-leak repair 的 joint_f1 仅 +0.0002；
- effective action rate 只有 0.7000；
- positive_vs_bm25_recall 只有 0.0433；
- 当前 1000 扩展大概率只会确认“与 BM25 持平或微弱波动”，成本高，论文收益低。

## 7. 下一步建议

优先级最高的下一步不是扩大样本，而是修 selector/candidate 的可识别性：

1. 先做 positive action detectability audit：只看 73 个 positive-vs-BM25 query，分析 positive action 与普通 action 的 feature margin。
2. 改善 candidate pool：当前 227 / 300 query 没有 positive-vs-BM25 action，说明候选动作生成本身比 selector 更硬。
3. 提高 positive recall，而不是追求全局激进选择：当前 answer-neutral selector 已证明保守策略能保 answer，但 positive recall 太低。
4. 论文中保留 2Wiki 作为 limitation / external sanity check，不把它写成主方法胜利。

## 8. 当前产物路径

最新主要产物：

- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/reports/2wiki_bm25_anchor_repair_report.md`
- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/reports/2wiki_external_validation_decision.md`
- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/outputs/selector_smoke_300/summary.json`
- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/outputs/ablation/ablation_summary.json`
- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/outputs/diagnostics/failure_summary.json`
- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/outputs/audit/no_leak_audit.md`

本报告建议同步为：

- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/reports/2wiki_current_progress_report_20260624.md`
- `V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor/reports/2wiki_current_progress_report_latest.md`
- `实验分析报告/V7-HP-PAPER/2wiki_current_progress_report_latest.md`
