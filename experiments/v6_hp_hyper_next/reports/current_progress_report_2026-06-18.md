# V6-HP-hyper 当前进展报告

- 报告时间：2026-06-18 18:50 JST
- 本地仓库：`/Users/iilab/PPRAG_FL/FedE4RAG-main/FedE4RAG-main`
- 服务器仓库：`/home/iiserver31/projects/FedE4RAG-main`
- 实验目录：`experiments/v6_hp_hyper_next/`

## 1. 当前状态概述

截至本次检查，Task A 的 HotpotQA hard query benchmark 已完成，subset 读取 smoke test 已完成，Task B 的第一步 same-payload baseline B1 也已完成。

当前没有 `v6_hp_hyper_next` 相关长实验进程在运行。服务器 GPU 基本空闲，但系统 load 较高，主要来自其它项目进程。`/home` 使用率约 90%，低于 95% 风险线，但后续多 seed 实验仍需持续监控磁盘。

## 2. 环境与资源状态

服务器检查时间：

```text
Thu Jun 18 18:49 JST 2026
```

磁盘状态：

```text
/home: 3.5T total, 3.2T used, 380G available, 90% used
```

GPU 状态：

| GPU | Used / Total | Utilization |
| --- | ---: | ---: |
| 0 | 23 MiB / 40960 MiB | 0% |
| 1 | 21 MiB / 40960 MiB | 0% |
| 2 | 21 MiB / 40960 MiB | 0% |
| 3 | 3596 MiB / 40960 MiB | 0% |

注意：系统 load 为 `57.5`，但这主要来自其它 CPU 任务；当前 GPU 不忙。

## 3. 已完成工作

### 3.1 Task A：HotpotQA hard query benchmark

Task A 已完成，不需要重复构建。正式可用文件包括：

```text
experiments/v6_hp_hyper_next/results/baseline_all1500_v3_per_query.jsonl
experiments/v6_hp_hyper_next/results/hard_subset_stats.csv
experiments/v6_hp_hyper_next/reports/hard_subset_stats.md
experiments/v6_hp_hyper_next/subsets/hotpot_all_1000.json
experiments/v6_hp_hyper_next/subsets/hotpot_hard_1000.json
experiments/v6_hp_hyper_next/subsets/hotpot_hard_500.json
```

关键统计：

| subset | hit@1 | hit@3 | answer_cov | support_cov | diff_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_1000 | 0.8420 | 0.9530 | 0.5530 | 0.9075 | 0.1523 |
| hard_1000 | 0.7660 | 0.9240 | 0.3280 | 0.8555 | 0.2332 |
| hard_500 | 0.6840 | 0.8480 | 0.1680 | 0.7110 | 0.3574 |

### 3.2 Task 0：subset 读取 smoke test

subset smoke test 已完成，报告文件：

```text
experiments/v6_hp_hyper_next/reports/subset_smoke_report.md
experiments/v6_hp_hyper_next/results/subset_smoke_raw.csv
```

结果：

| case | per_query_lines | F1 | EM | MRR | NDCG | Recall@3 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke_all_50 | 50 | 0.8800 | 0.8000 | 0.9500 | 0.8997 | 0.8900 | PASS |
| smoke_hard_50 | 50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS as stress subset |

说明：

- `--query-subset` 可以正确读取新的 `queries` 字段结构。
- 日志确认使用 `hotpot_split=validation` 和 `hotpot_examples=1500`。
- 没有退回旧 `json_download` 金融数据。
- `smoke_hard_50` 全 0 是因为选取的是 hard_500 中最难的前 50 条，属于极端 stress smoke，不代表 hard_500 全集平均水平。

### 3.3 Task B1：same-payload baseline on all_1000 / seed=42

B1 已完成。运行范围：

```text
subset = hotpot_all_1000
seed = 42
methods = V3, V4, V5, V6-HP-hyper anchor
target_payload = 0.070134 ± 0.002
```

产物：

```text
experiments/v6_hp_hyper_next/results/same_payload_baseline_raw.csv
experiments/v6_hp_hyper_next/results/same_payload_baseline_summary.csv
experiments/v6_hp_hyper_next/reports/same_payload_baseline_report.md
experiments/v6_hp_hyper_next/logs/same_payload_b1_all1000_20260617_194351.nohup.log
```

B1 结果：

| Method | Payload | MRR | NDCG | F1 | EM | Recall@3 | Hit@10 | Payload OK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V3_topk2_fixed | 0.070134 | 0.8923 | 0.8745 | 0.7985 | 0.7320 | 0.9100 | 0.9550 | yes |
| V4_topk2_fixed | 0.070134 | 0.8923 | 0.8745 | 0.7985 | 0.7320 | 0.9100 | 0.9550 | yes |
| V5_topk2_fixed | 0.070134 | 0.8923 | 0.8745 | 0.7985 | 0.7320 | 0.9100 | 0.9550 | yes |
| V6_HP_hyper_anchor | 0.070134 | 0.8930 | 0.8745 | 0.7990 | 0.7310 | 0.9090 | 0.9540 | yes |

## 4. 当前数据分析

### 4.1 同 payload 校准成功

四个方法的 payload 都为约：

```text
0.070134334
```

均落在目标区间 `0.070134 ± 0.002` 内。因此 B1 是有效的 same-payload 对比，不存在 V6 用更高通信预算获得优势的问题。

### 4.2 all_1000 仍然拉不开明显差距

在 `hotpot_all_1000 / seed=42` 上，V6-HP-hyper anchor 相比 V3/V4/V5：

| Metric | V6 - baseline |
| --- | ---: |
| MRR | +0.0007 |
| F1 | +0.0005 |
| NDCG | 0.0000 |
| EM | -0.0010 |
| Recall@3 | -0.0010 |
| Hit@10 | -0.0010 |

这个结果说明 V6 能够在严格低 payload 下保持性能，但目前不能凭 `all_1000` 单 seed 结果宣称 V6 明显优于 V3/V4/V5。

### 4.3 更关键的验证应转向 hard split

由于 `all_1000` 上差距极小，下一步最有价值的是运行：

```text
hotpot_hard_1000 / seed=42
hotpot_hard_500 / seed=42
```

如果 V6 的优势存在，更可能在 `hard_1000` 或 `hard_500` 上体现，而不是在标准 `all_1000` 上体现。

## 5. 当前风险与注意事项

1. `/home` 当前约 90%，尚可继续小规模实验，但不建议同时启动多个大矩阵。
2. 服务器 CPU load 较高，可能影响 CPU-heavy downstream eval 或日志响应速度。
3. B1 恢复过 V3/V4/V5 代码文件，后续如果要同步 GitHub，需要注意服务器 git status 中仍有大量历史删除/变更，不应盲目全量提交。
4. 当前 B1 只是单 seed、单 split，不具备统计显著性。

## 6. 下一步计划

推荐下一步按以下顺序推进：

1. 启动 Task B2：`hotpot_hard_1000 / seed=42 / V3,V4,V5,V6`。
2. 若 B2 正常，再跑 Task B3：`hotpot_hard_500 / seed=42 / V3,V4,V5,V6`。
3. 比较 all_1000、hard_1000、hard_500 三个 split 后，再决定是否扩展到 seeds 43/44。
4. 如果 hard split 仍拉不开差距，则优先做 V6 ablation，验证 layerwise budget 是否是稳定性来源。
5. adaptive same-payload verification 应放在 hard split 结果之后，避免在不敏感 split 上浪费计算。

## 7. 当前结论

当前最稳妥的结论是：

```text
Under strict same-payload constraints on all_1000, V6-HP-hyper preserves FedRAG retrieval performance at payload≈0.0701, but does not yet show a practically large advantage over V3/V4/V5. The next decisive test should be hard_1000 and hard_500.
```

Adaptive budget 仍不能宣称已被独立验证：

```text
Adaptive budget is not yet independently validated under strict same-payload constraints.
```
