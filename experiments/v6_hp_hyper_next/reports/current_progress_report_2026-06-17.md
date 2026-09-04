# V6-HP-hyper 下一阶段实验当前进展报告

- 报告时间：2026-06-17 19:30 JST
- 本地仓库：`/Users/iilab/PPRAG_FL/FedE4RAG-main/FedE4RAG-main`
- 服务器仓库：`/home/iiserver31/projects/FedE4RAG-main`
- 实验目录：`experiments/v6_hp_hyper_next/`
- 当前阶段状态：Task A 的 HotpotQA hard query benchmark 已从昨日阻塞状态推进为可用状态；same-payload baseline、V6 ablation、adaptive same-payload verification 尚未正式启动。

## 1. 实验目的

本阶段实验的目标是为 FedRAG / FL-RAG 论文级实验建立更可靠的评测证据，而不是继续盲目增加机制复杂度。前期 V6-HP1 与 V6-HP1-OPTUNA 已经找到一个稳定的低通信 anchor：

| 参数 | 设置 |
| --- | --- |
| `topk` | 2 |
| `warmup` | 0 |
| `budget_mode` | fixed |
| `score_mode` | value |
| `use_utility_memory` | False |
| `layerwise_budget` | True |
| payload | 约 0.070134 |

已有 Optuna 结果显示该配置在当前 HotpotQA proxy 上可以取得较好表现，但也暴露了三个核心问题：proxy 指标有饱和趋势、hard query subset 之前只有 50 条、adaptive budget 尚未在同等 payload 下独立超过 fixed low-budget。因此，本阶段的核心是把评测基准和对照实验做严谨。

本阶段要回答的问题包括：

1. 在 payload≈0.070134±0.002 下，V6-HP-hyper 是否优于 V3/V4/V5？
2. V6 的优势是否在 hard query subset 上更明显？
3. `layerwise_budget` 是否是低预算稳定性的关键因素？
4. adaptive budget 在不增加总 payload 时是否真的优于 fixed low-budget？
5. utility memory 是否应保留，还是仅作为短期/tie-breaker 辅助机制？

## 2. 当前服务器状态

本次查询服务器状态如下：

| 项目 | 状态 |
| --- | --- |
| `v6_hp_hyper_next` 相关进程 | 未发现正在运行的长实验进程 |
| GPU 0 | 23 MiB / 40960 MiB, 0% utilization |
| GPU 1 | 21 MiB / 40960 MiB, 0% utilization |
| GPU 2 | 21 MiB / 40960 MiB, 0% utilization |
| GPU 3 | 3596 MiB / 40960 MiB, 0% utilization |
| `/home` 使用率 | 97.4% of 3.49TB |
| `experiments/v6_hp_hyper_next` 大小 | 286M |
| `V6-HP1/outputs` 大小 | 112G |
| `V6-HP1-OPTUNA/outputs` 大小 | 17M |

GPU 资源目前空闲，但磁盘空间已经非常紧张。后续 full validation、大规模 baseline 或多 seed ablation 启动前，需要优先控制输出规模，必要时清理或转移旧的大型中间产物。

## 3. 关键进展

### 3.1 HotpotQA 数据加载问题已修复

昨日发现 `baseline_all1500` 和 `baseline_all1500_v2` 虽然外层传入了：

```text
RAGTEST_DATASET=hotpot_qa
HOTPOT_SPLIT=validation
HOTPOT_MAX_EXAMPLES=1500
```

但服务器 `RAGTest/config.py` 未应用环境变量 override，导致内部实际仍使用 `json_download`，最终只生成 50 条金融 QA per-query。该问题已经通过同步本地正确版 `RAGTest/config.py` 修复。

修复后 smoke test 已确认：

```text
dataset=hotpot_qa
hotpot_split=validation
hotpot_examples=5
```

随后启动的 `baseline_all1500_v3` 已完成，确认使用 HotpotQA validation，并生成 1500 行 per-query 结果。

### 3.2 Corrected HotpotQA per-query 生成完成

正式可用的 per-query 文件为：

```text
experiments/v6_hp_hyper_next/results/baseline_all1500_v3_per_query.jsonl
```

行数：

```text
1500
```

日志中的最终指标如下：

| Metric | Value |
| --- | ---: |
| F1 | 0.800333 |
| EM | 0.734000 |
| MRR | 0.891444 |
| Hit@1 | 0.919333 |
| Hit@10 | 0.949333 |
| NDCG | 0.871922 |
| Recall@1 | 0.6867 |
| Recall@3 | 0.9037 |
| Recall@5 | 0.9037 |
| Recall@10 | 0.9037 |
| Gold rank | 1.0860 |

该结果和旧的 50 条金融 QA 输出相比，已经是正确的 HotpotQA validation proxy，可作为重建 hard query benchmark 的输入。

### 3.3 Difficulty-ranked hard query benchmark 已完成重建

已使用正确参数运行：

```bash
/home/iiserver31/anaconda3/envs/supv2/bin/python experiments/v6_hp_hyper_next/build_ranked_hotpot_subsets.py \
  --per-query-input experiments/v6_hp_hyper_next/results/baseline_all1500_v3_per_query.jsonl \
  --output-dir experiments/v6_hp_hyper_next/subsets \
  --results-csv experiments/v6_hp_hyper_next/results/hard_subset_stats.csv \
  --report-md experiments/v6_hp_hyper_next/reports/hard_subset_stats.md
```

生成的 subset 均已达到目标规模：

| Subset | Count |
| --- | ---: |
| `hotpot_easy_1000.json` | 1000 |
| `hotpot_medium_1000.json` | 1000 |
| `hotpot_hard_500.json` | 500 |
| `hotpot_hard_1000.json` | 1000 |
| `hotpot_all_1000.json` | 1000 |
| `hotpot_full_eval.json` | 1500 |

对应统计如下：

| subset | num_examples | avg_gold_rank | hit@1 | hit@3 | hit@10 | answer_cov | support_cov | diff_mean | diff_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpot_easy_1000 | 1000 | 1.0940 | 0.9240 | 1.0000 | 1.0000 | 0.7440 | 1.0000 | 0.0545 | 0.0862 |
| hotpot_medium_1000 | 1000 | 1.0940 | 0.9240 | 1.0000 | 1.0000 | 0.5060 | 0.9940 | 0.1048 | 0.0977 |
| hotpot_hard_500 | 500 | 1.0700 | 0.6840 | 0.8480 | 0.8480 | 0.1680 | 0.7110 | 0.3574 | 0.2712 |
| hotpot_hard_1000 | 1000 | 1.1290 | 0.7660 | 0.9240 | 0.9240 | 0.3280 | 0.8555 | 0.2332 | 0.2380 |
| hotpot_all_1000 | 1000 | 1.0920 | 0.8420 | 0.9530 | 0.9530 | 0.5530 | 0.9075 | 0.1523 | 0.2166 |
| hotpot_full_eval | 1500 | 1.0860 | 0.8440 | 0.9493 | 0.9493 | 0.5520 | 0.9037 | 0.1555 | 0.2233 |

这说明 Task A 的关键阻塞已经解除：hard subset 不再只有 50 条，而是可以正式支持 `hard_500` 与 `hard_1000` 评估。

## 4. 当前数据分析

### 4.1 Hard subset 的判别力明显增强

与 `all_1000` 和 `full_eval` 相比，`hard_500` 的难度明显更高：

| 对比项 | all_1000 | hard_500 | 变化 |
| --- | ---: | ---: | ---: |
| hit@1 | 0.8420 | 0.6840 | -0.1580 |
| hit@3 | 0.9530 | 0.8480 | -0.1050 |
| answer_cov | 0.5530 | 0.1680 | -0.3850 |
| support_cov | 0.9075 | 0.7110 | -0.1965 |
| diff_mean | 0.1523 | 0.3574 | +0.2051 |

这比之前 50 条 subset 更有实验价值。尤其 `hard_500` 的 answer coverage 只有 0.1680，说明该 split 更容易暴露检索不足或预算选择失败，对验证 V6 的 low-budget selective upload 更有判别力。

### 4.2 `hard_1000` 适合作为主 hard split，`hard_500` 适合作为 stress split

`hard_1000` 相比 `hard_500` 更大、更稳定，但难度被中等困难样本稀释：

| 指标 | hard_500 | hard_1000 |
| --- | ---: | ---: |
| hit@1 | 0.6840 | 0.7660 |
| hit@3 | 0.8480 | 0.9240 |
| answer_cov | 0.1680 | 0.3280 |
| diff_mean | 0.3574 | 0.2332 |
| diff_std | 0.2712 | 0.2380 |

建议后续论文图表使用：

1. `all_1000` 作为主标准 validation proxy。
2. `hard_1000` 作为主要 hard-query evaluation。
3. `hard_500` 作为 stress-test 或补充表格。

### 4.3 当前仍不能宣称 adaptive 已有效

虽然 hard benchmark 已经重建，但 same-payload baseline、V6 ablation、adaptive same-payload verification 尚未运行。因此当前仍只能支持：

```text
low-budget fixed anchor 是目前最稳定的已验证配置。
```

还不能支持：

```text
adaptive budget 在同等 payload 下显著优于 fixed low-budget。
```

## 5. 已同步到本地的文件

以下服务器新产物已经同步回本地：

```text
experiments/v6_hp_hyper_next/results/baseline_all1500_v3_per_query.jsonl
experiments/v6_hp_hyper_next/results/hard_subset_stats.csv
experiments/v6_hp_hyper_next/reports/hard_subset_stats.md
experiments/v6_hp_hyper_next/subsets/hotpot_easy_1000.json
experiments/v6_hp_hyper_next/subsets/hotpot_medium_1000.json
experiments/v6_hp_hyper_next/subsets/hotpot_hard_500.json
experiments/v6_hp_hyper_next/subsets/hotpot_hard_1000.json
experiments/v6_hp_hyper_next/subsets/hotpot_all_1000.json
experiments/v6_hp_hyper_next/subsets/hotpot_full_eval.json
```

本地命令日志也已更新：

```text
experiments/v6_hp_hyper_next/logs/all_commands.log
```

## 6. 后续 Todo List

### 6.1 立即建议

1. 在启动大规模后续实验前，先处理服务器磁盘压力，尤其是 `V6-HP1/outputs` 约 112G。
2. 将当前正式 hard subset 作为 Task B/C/D 的固定评测输入，避免再次使用旧 50 条结果。
3. 运行一个小规模 smoke benchmark，确认 `--query-subset` 可以正确读取新的 subset JSON 结构。

### 6.2 Task B：Same-payload baseline benchmark

目标是在 payload≈0.070134±0.002 下比较：

```text
V3
V4
V5
V6-HP-hyper best
```

建议优先顺序：

1. 先跑 `all_1000` 单 seed smoke。
2. 再跑 `hard_500` 单 seed smoke。
3. 确认 subset 读取与 payload 记录正确后，扩展到 seeds 42/43/44。
4. 输出 `same_payload_baseline_raw.csv`、`same_payload_baseline_summary.csv`、`same_payload_baseline_report.md`。

### 6.3 Task C：V6 ablation

围绕 anchor 运行：

```text
v6_fixed_anchor
v6_no_layerwise
v6_delta_score
v6_utility_memory_full
v6_utility_memory_ema
v6_hard_weighting_off
v6_hard_weighting_on
```

重点观察 `hard_500` 与 `hard_1000` 上的 MRR/F1/EM 是否真正拉开差距。

### 6.4 Task D：Adaptive same-payload verification

后续必须严格记录实际 payload，比较：

```text
fixed_anchor
adaptive_realloc_same_payload
adaptive_capped_same_payload
adaptive_v6_original
```

如果 adaptive 只在 payload 更高时提升，则不能作为主论文 claim。

## 7. 当前结论

截至 2026-06-17，V6-HP-hyper 下一阶段实验已经完成最关键的评测基准修复：正确的 HotpotQA validation per-query 数据已经生成 1500 条，difficulty-ranked hard query benchmark 已经重建，并成功得到 `hard_500` 与 `hard_1000`。

当前最重要的进展是：之前“hard subset 只有 50 条”的瓶颈已经解决，下一阶段可以正式进入 same-payload baseline、V6 ablation 和 adaptive same-payload verification。

当前最稳妥的结论仍然是：

```text
Under strict same-payload constraints, the strongest supported claim is: low-budget selective upload with layerwise budget can preserve FedRAG retrieval performance at payload≈0.0701; adaptive budget remains promising but is not yet independently validated.
```
