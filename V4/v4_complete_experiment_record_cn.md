# V4 完整实验结果记录

## 1. 实验目标

V4 的目标是在 V3 的基础上进一步验证一条更强的假设：

1. 上游联邦训练中的上传选择不仅可以基于参数重要性和通信成本；
2. 还可以进一步引入 downstream-aware value、hard-query / hard-client 感知和 utility memory；
3. 从而在相同或接近的通信预算下，更稳定地保持下游 RAG 检索效果。

## 2. 代码与运行位置

- 本地代码目录：`/Users/iilab/PPRAG_FL/FedE4RAG-main/FedE4RAG-main/V4`
- 服务器代码目录：`/home/iiserver31/projects/FedE4RAG-main/V4`
- 自动化总控脚本：`/home/iiserver31/projects/FedE4RAG-main/run_v4_all.sh`
- 服务器上游输出目录：`/home/iiserver31/projects/FedE4RAG-main/V4/outputs/pprag_fl_v4`
- 服务器下游输出目录：`/home/iiserver31/projects/FedE4RAG-main/V4/outputs/rag_eval_all_v4`
- 服务器报告目录：`/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V4`

## 3. 自动化执行方式

V4 全流程使用 `run_v4_all.sh` 自动执行，包含：

1. 串行执行上游 suite：
   - `v4_main`
   - `v4_budget`
   - `v4_heterogeneity`
   - `v4_hardquery`
   - `v4_ablation_signal`
   - `v4_ablation_budget`
   - `v4_explain`
2. 每套 suite 结束后自动执行 `V4/finalize_pipeline.py`
3. 自动补跑对应下游 `RAGTest`
4. 自动生成 `full_pipeline_*`
5. 最终执行 `all_v4 finalize`

## 4. 完成情况

### 4.1 总体完成规模

- 上游实验总数：`121`
- 下游评测总数：`121`
- 上下游闭环完成：`121 / 121`

### 4.2 各 suite 完成情况

| Suite | 完成情况 | 说明 |
|---|---:|---|
| `smoke` | 1 | 冒烟验证 |
| `v4_main` | 15 | 主结果对比 |
| `v4_budget` | 27 | 同预算/不同预算点对比 |
| `v4_heterogeneity` | 48 | 不同异构强度对比 |
| `v4_hardquery` | 6 | 困难 query 条件下对比 |
| `v4_ablation_signal` | 12 | downstream-aware signal / history / client 等消融 |
| `v4_ablation_budget` | 9 | fixed / adaptive / adaptive_v4 等预算机制消融 |
| `v4_explain` | 3 | explain 专用实验 |

注：`smoke + 15 + 27 + 48 + 6 + 12 + 9 + 3 = 121`。

## 5. 已生成报告

### 5.1 Suite 报告

- `suite_smoke_2026-05-18_16-42-57`
- `suite_v4_main_2026-05-18_23-58-02`
- `suite_v4_budget_2026-05-19_09-24-49`
- `suite_v4_heterogeneity_2026-05-20_06-57-19`
- `suite_v4_hardquery_2026-05-20_08-13-55`
- `suite_v4_ablation_signal_2026-05-20_10-32-25`
- `suite_v4_ablation_budget_2026-05-20_12-18-18`
- `suite_v4_explain_2026-05-20_12-54-47`

### 5.2 Full Pipeline 报告

- `full_pipeline_v4_main_2026-05-19_00-00-40`
- `full_pipeline_v4_budget_2026-05-19_09-29-00`
- `full_pipeline_v4_heterogeneity_2026-05-20_07-05-01`
- `full_pipeline_v4_hardquery_2026-05-20_08-14-51`
- `full_pipeline_v4_ablation_signal_2026-05-20_10-34-17`
- `full_pipeline_v4_ablation_budget_2026-05-20_12-19-42`
- `full_pipeline_v4_explain_2026-05-20_12-55-16`
- `full_pipeline_all_v4_2026-05-20_12-55-42`

## 6. 主结果记录

### 6.1 `v4_main`

| 方法 | 平均 payload | 平均 reduction | cos_3 | recall_3 | MRR | NDCG |
|---|---:|---:|---:|---:|---:|---:|
| `full` | `1.0000` | `0.0000` | `0.9600` | `0.9433` | `0.9200` | `0.9132` |
| `random` | `0.2575` | `0.7425` | `0.9600` | `0.9433` | `0.9089` | `0.9049` |
| `delta_norm` | `0.2316` | `0.7684` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `hypernet_v3` | `0.2266` | `0.7734` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `hypernet_v4` | `0.2822` | `0.7178` | `0.9600` | `0.9433` | `0.9000` | `0.8985` |

### 6.2 `v4_budget`

#### 代表性预算点

| 方法 | 配置 | 平均 payload | cos_3 | recall_3 | MRR | NDCG |
|---|---|---:|---:|---:|---:|---:|
| `random` | `k=1` | `0.1095` | `0.9600` | `0.9333` | `0.9067` | `0.8971` |
| `delta_norm` | `k=1` | `0.1073` | `0.9600` | `0.9333` | `0.9067` | `0.8939` |
| `hypernet_v3` | `k=1` | `0.1028` | `0.9600` | `0.9333` | `0.9067` | `0.8939` |
| `hypernet_v4` | `k=1` | `0.1596` | `0.9600` | `0.9333` | `0.9067` | `0.8955` |
| `random` | `k=3` | `0.2575` | `0.9600` | `0.9433` | `0.9089` | `0.9049` |
| `delta_norm` | `k=3` | `0.2316` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `hypernet_v3` | `k=3` | `0.2266` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `hypernet_v4` | `k=3` | `0.2822` | `0.9600` | `0.9433` | `0.9000` | `0.8985` |
| `hypernet_v4` | `k=5` | `0.4025` | `0.9600` | `0.9433` | `0.9100` | `0.9058` |

### 6.3 `v4_heterogeneity`

#### 不同异构强度下的代表性结果

| 任务 | 方法 | 平均 payload | cos_3 | recall_3 | MRR | NDCG |
|---|---|---:|---:|---:|---:|---:|
| `alpha=0.5` | `random` | `0.2575` | `0.9600` | `0.9433` | `0.9089` | `0.9049` |
| `alpha=0.5` | `delta_norm` | `0.2316` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `alpha=0.5` | `hypernet_v3` | `0.2258` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `alpha=0.5` | `hypernet_v4` | `0.2819` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `alpha=0.3` | `random` | `0.2575` | `0.9600` | `0.9433` | `0.9089` | `0.9049` |
| `alpha=0.3` | `delta_norm` | `0.2316` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `alpha=0.3` | `hypernet_v3` | `0.2266` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `alpha=0.3` | `hypernet_v4` | `0.2822` | `0.9600` | `0.9433` | `0.9000` | `0.8985` |
| `alpha=0.1` | `random` | `0.2575` | `0.9600` | `0.9433` | `0.9078` | `0.9041` |
| `alpha=0.1` | `delta_norm` | `0.2316` | `0.9600` | `0.9433` | `0.9100` | `0.9058` |
| `alpha=0.1` | `hypernet_v3` | `0.2261` | `0.9600` | `0.9433` | `0.9100` | `0.9058` |
| `alpha=0.1` | `hypernet_v4` | `0.2817` | `0.9600` | `0.9433` | `0.9167` | `0.9107` |
| `alpha=0.05` | `delta_norm` | `0.2316` | `0.9600` | `0.9533` | `0.9200` | `0.9194` |
| `alpha=0.05` | `hypernet_v3` | `0.2259` | `0.9600` | `0.9466` | `0.9133` | `0.9103` |
| `alpha=0.05` | `hypernet_v4` | `0.2826` | `0.9600` | `0.9533` | `0.9200` | `0.9194` |

### 6.4 `v4_hardquery`

| 方法 | 平均 payload | cos_3 | recall_3 | MRR | NDCG |
|---|---:|---:|---:|---:|---:|
| `hypernet_v3` | `0.2268` | `0.9600` | `0.9433` | `0.8967` | `0.8958` |
| `hypernet_v4` | `0.2822` | `0.9600` | `0.9433` | `0.9000` | `0.8985` |

### 6.5 `v4_ablation_signal`

分组统计：

- 平均 payload：`0.2821 ± 0.0018`
- 平均 reduction：`0.7179 ± 0.0018`
- `cos_3 = 0.9600`
- `recall_3 = 0.9433`
- `MRR = 0.9000`
- `NDCG = 0.8985`

### 6.6 `v4_ablation_budget`

分组统计：

- 平均 payload：`0.2452 ± 0.0263`
- 平均 reduction：`0.7548 ± 0.0263`
- `cos_3 = 0.9600`
- `recall_3 = 0.9433`
- `MRR = 0.8978`
- `NDCG = 0.8967`

### 6.7 `v4_explain`

- 平均 payload：`0.3733 ± 0.0000`
- 平均 reduction：`0.6267 ± 0.0000`
- `cos_3 = 0.9600`
- `recall_3 = 0.9400 ± 0.0047`
- `MRR = 0.9067`
- `NDCG = 0.9006 ± 0.0036`

## 7. 阶段性记录结论

1. V4 全流程已经稳定跑通，`121` 个上游与 `121` 个下游结果全部闭环。
2. V4 在主实验、预算实验、异构性实验、hardquery、signal/budget 消融和 explain 套件中都完成了可复查结果。
3. 当前最明显的行为变化是：`adaptive_v4` 在更难或更异构的场景中，会把上传预算由 `3` 个块提高到 `4` 个块。
4. 从最终汇总结果看，V4 的下游指标总体稳定，但其通信代价普遍高于 V3。

## 8. 关键文件与路径

- 总汇总报告：
  - `/home/iiserver31/projects/FedE4RAG-main/实验分析报告/V4/full_pipeline_all_v4_2026-05-20_12-55-42/report.md`
- 上游总汇总：
  - `/home/iiserver31/projects/FedE4RAG-main/V4/outputs/pprag_fl_v4/summary.json`
  - `/home/iiserver31/projects/FedE4RAG-main/V4/outputs/pprag_fl_v4/summary_grouped.json`
- 下游总输出：
  - `/home/iiserver31/projects/FedE4RAG-main/V4/outputs/rag_eval_all_v4`
