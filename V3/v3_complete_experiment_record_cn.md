# V3 完整实验结果记录

## 1. 实验目标

V3 的目标是在 V2“超网络选择性上传”框架基础上，进一步引入：

- 客户端条件特征
- 历史记忆特征
- value-aware 打分
- 自适应预算分配

从而在相同通信预算下进一步降低上游通信量，并尽可能保持下游 RAG 检索效果稳定。

## 2. 实验平台与输出位置

- 本地代码目录：`/Users/iilab/PPRAG_FL/FedE4RAG-main/FedE4RAG-main/V3`
- 服务器实验目录：`/home/iiserver31/projects/FedE4RAG-main/V3`
- 上游输出目录：`V3/outputs/pprag_fl_v3/`
- 下游输出目录：`V3/outputs/rag_eval_all_v3/`
- 报告归档目录：`实验分析报告/V3/`

## 3. 实验套件与规模

本轮 V3 自动化总控脚本最终完成了以下实验套件：

| 套件 | 设计用途 | 完成数量 |
|---|---:|---:|
| `smoke` | 流程验证 | 1 |
| `v3_main` | 主结果对比 | 15 |
| `v3_budget` | 不同预算点对比 | 27 |
| `v3_heterogeneity` | 不同异构程度对比 | 36 |
| `v3_ablation_feature` | 特征消融 | 12 |
| `v3_ablation_budget` | 预算机制消融 | 6 |
| `v3_explain` | 解释性实验 | 3 |

说明：

- 全流程总报告统计到的已完成上游实验数为 `100`，已完成下游实验数也为 `100`。
- `v3_ablation_budget` 的 suite manifest 按设计有 9 个配置，但最终有效落盘 run 为 6 个；当前结果应以 `full_pipeline_all_v3` 的实际归档数量为准。

## 4. 已生成的关键报告

### 4.1 Suite 报告

- `suite_smoke_2026-05-11_14-42-27`
- `suite_v3_main_2026-05-13_21-06-26`
- `suite_v3_budget_2026-05-15_13-25-12`
- `suite_v3_heterogeneity_2026-05-16_15-13-10`
- `suite_v3_ablation_feature_2026-05-17_05-05-57`
- `suite_v3_ablation_budget_2026-05-18_00-51-58`
- `suite_v3_explain_2026-05-18_02-36-08`

### 4.2 Full Pipeline 报告

- `full_pipeline_v3_main_2026-05-17_21-15-06`
- `full_pipeline_v3_budget_2026-05-17_21-20-17`
- `full_pipeline_v3_heterogeneity_2026-05-17_21-27-14`
- `full_pipeline_v3_ablation_feature_2026-05-17_21-29-26`
- `full_pipeline_v3_ablation_budget_2026-05-18_00-54-30`
- `full_pipeline_v3_explain_2026-05-18_02-36-39`
- `full_pipeline_all_v3_2026-05-18_02-53-03`

## 5. 关键上游结果记录

### 5.1 主结果 `v3_main`

| 方法 | 平均上传比例 | 平均压缩率 |
|---|---:|---:|
| `full` | `1.0000` | `0.0000` |
| `random(k=3)` | `0.2575` | `0.7425` |
| `delta_norm(k=3)` | `0.2377` | `0.7623` |
| `hypernet_v2(k=3)` | `0.2316` | `0.7684` |
| `hypernet_v3(k=3)` | `0.2259` | `0.7741` |

主结果中，`hypernet_v3` 获得了最小平均上传比例。

### 5.2 预算实验 `v3_budget`

| 方法 | 配置 | 平均上传比例 | 平均压缩率 |
|---|---|---:|---:|
| `random` | `k=1` | `0.1095` | `0.8905` |
| `delta_norm` | `k=1` | `0.1073` | `0.8927` |
| `hypernet_v2` | `k=1` | `0.1073` | `0.8927` |
| `hypernet_v3` | `k=1, adaptive` | `0.1020` | `0.8980` |
| `random` | `k=3` | `0.2575` | `0.7425` |
| `delta_norm` | `k=3` | `0.2431` | `0.7569` |
| `hypernet_v2` | `k=3` | `0.2316` | `0.7684` |
| `hypernet_v3` | `k=3, adaptive` | `0.2263` | `0.7737` |
| `hypernet_v3` | `k=5, adaptive` | `0.3489` | `0.6511` |

预算实验已经清晰展示出不同 `top-k` 对上传比例的分层影响。

### 5.3 异构性实验 `v3_heterogeneity`

在 `topk=3` 下，主要结果如下：

| 场景 | `random` | `delta_norm` | `hypernet_v2` | `hypernet_v3` |
|---|---:|---:|---:|---:|
| `alpha=0.5` | `0.2575` | `0.2418` | `0.2316` | `0.2244` |
| `alpha=0.3` | `0.2575` | `0.2431` | `0.2316` | `0.2263` |
| `alpha=0.1` | `0.2575` | `0.3042` | `0.2316` | `0.2256` |

### 5.4 特征消融 `v3_ablation_feature`

| 配置 | 平均上传比例 | 平均压缩率 |
|---|---:|---:|
| `value + history + client` | `0.2263` | `0.7737` |
| `value + no history` | `0.2233` | `0.7767` |
| `value + no client embedding` | `0.2299` | `0.7701` |
| `importance + history + client` | `0.2345` | `0.7655` |

### 5.5 预算机制消融 `v3_ablation_budget`

| 配置 | 平均上传比例 | 平均压缩率 | 预测预算比例 |
|---|---:|---:|---:|
| `adaptive` | `0.2263` | `0.7737` | `0.4838` |
| `fixed` | `0.2263` | `0.7737` | `0.0000` |

## 6. 下游评测记录

- `full_pipeline_all_v3` 记录到已完成下游实验数：`100`
- 当前归档中 `rag_eval_stdout.log` 数量：`199`
- 代表性下游指标在多数配置中表现接近，常见取值区间包括：
  - `cos_1 ≈ 0.84 ~ 0.88`
  - `cos_3 ≈ 0.94 ~ 0.96`
  - `recall_3 ≈ 0.9133 ~ 0.9433`
  - `mrr ≈ 0.8967 ~ 0.9200`
  - `NDCG ≈ 0.8855 ~ 0.9132`

## 7. 结果记录中的重要说明

1. 当前 V3 已完成上游、下游和全局归档，结果已经进入可分析阶段。
2. `v3_ablation_budget` 的 suite 报告显示 9 个运行配置，但全局归档中的有效上游 run 数为 6；这是解释结果时必须说明的限制。
3. 下游指标在大量配置间差异较小，说明当前检索评测设置下，方法差异主要先体现在上游通信层面。

## 8. 最终归档基准

本轮整理以以下全局总报告为准：

- `实验分析报告/V3/full_pipeline_all_v3_2026-05-18_02-53-03/`

其中包含：

- `report.md`
- `report.json`
- `data/upstream_summary.json`
- `data/upstream_summary_grouped.json`
- `data/downstream_summary.json`

后续论文写作和 PPT 汇报，均建议以该总归档为主来源。
