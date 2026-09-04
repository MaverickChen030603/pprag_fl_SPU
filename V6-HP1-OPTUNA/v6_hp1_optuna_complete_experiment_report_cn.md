# V6-HP1-OPTUNA 完整实验报告

- 整理时间：2026-06-12 16:37:07 JST
- 实验位置：`/home/iiserver31/projects/FedE4RAG-main/V6-HP1-OPTUNA`
- 关联实验：`V6-HP1` HotpotQA 难下游任务实验
- 搜索方式：Optuna TPE sampler，小规模 20-trial 参数搜索

## 1. 实验目的

本实验是在 `V6-HP1` 引入 HotpotQA 后追加的自动化调参实验，目标不是重新提出一个新方法，而是检验 V6 方法中的预算控制、hard query/hard client 信号、utility memory、layerwise budget 等设计，在更敏感的 HotpotQA 下游评价中是否存在更优组合。

具体问题包括：

- 在固定小规模训练预算下，哪些 V6 参数组合能取得更高的下游 utility proxy？
- `adaptive_v6` 是否优于更简单的 `fixed` 预算策略？
- 更大的 `topk`、warmup、utility memory 或 hard query weighting 是否能带来下游收益？
- 在通信成本受限时，是否存在更合适的低 payload 配置，能够作为后续正式实验候选。

## 2. 实验配置

| 项目 | 设置 |
| --- | --- |
| 数据集 | HotpotQA validation proxy |
| 训练数据 | `FedE/select_data_hotpot_train_5000.json` |
| 每个 trial 上游训练 | `hypernet_v6`, 5 clients, 10 rounds, 1 epoch, batch size 1 |
| 每个 trial 下游评估 | HotpotQA validation 300 examples |
| 优化器/搜索器 | Optuna TPE sampler, multivariate=True |
| 目标函数 | 0.30*mrr + 0.25*ndcg + 0.20*f1 + 0.15*em + 0.10*recall_3 - 0.25*payload |
| 计划 trial 数 | 20 |
| 实际记录 trial 数 | 21，其中 completed=20，failed=1 |
| 输出目录 | `V6-HP1-OPTUNA/outputs` |

搜索空间覆盖 `topk`、`warmup`、`score_mode`、`budget_mode`、hard query/client 阈值、adaptive expand/shrink 阈值、history window、hard query weighting、utility memory 和 layerwise budget。

## 3. 总体结果

| 指标 | 结果 |
| --- | --- |
| 完成有效 trial | 20 |
| 失败 trial | 1 |
| 最佳 trial | trial_0002 |
| 最佳 objective | 0.832511 |
| 最佳 payload | 0.070134 |
| 最佳 MRR | 0.9000 |
| 最佳 NDCG | 0.8855 |
| 最佳 F1 | 0.7967 |
| 最佳 EM | 0.7200 |
| 最佳 Recall@3 | 0.9133 |

最佳结果为 `trial_0002`，但后续多个 trial 复现了完全相同的 objective 与 downstream metrics，说明当前 search 的最优区域较集中，主要由 `topk=2 + warmup=0 + layerwise_budget=True + utility_memory=False` 这类低预算配置主导。

## 4. 最佳参数

| 参数 | 最佳值 |
| --- | --- |
| budget_mode | fixed |
| topk | 2 |
| warmup | 0 |
| score_mode | value |
| history_window | 5 |
| use_hard_query_weighting | True |
| use_utility_memory | False |
| layerwise_budget | True |

最佳 trial 的下游指标为：

| metric | value |
| --- | --- |
| MRR | 0.9000 |
| NDCG | 0.8855 |
| F1 | 0.7967 |
| EM | 0.7200 |
| Recall@3 | 0.9133 |
| Hit@1 | 0.8800 |
| Hit@10 | 0.9400 |

## 5. Trial 排名

| rank | trial | objective | payload | topk | warmup | budget | score | utility_memory | layerwise | MRR | NDCG | F1 | EM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0002 | 0.832511 | 0.0701 | 2 | 0 | fixed | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 2 | 0011 | 0.832511 | 0.0701 | 2 | 0 | fixed | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 3 | 0012 | 0.832511 | 0.0701 | 2 | 0 | fixed | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 4 | 0016 | 0.832511 | 0.0701 | 2 | 0 | adaptive_v6 | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 5 | 0017 | 0.832511 | 0.0701 | 2 | 0 | fixed | downstream_value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 6 | 0018 | 0.832511 | 0.0701 | 2 | 0 | fixed | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 7 | 0019 | 0.832511 | 0.0701 | 2 | 0 | fixed | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 8 | 0020 | 0.832511 | 0.0701 | 2 | 0 | fixed | downstream_value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 9 | 0014 | 0.809265 | 0.1631 | 2 | 1 | fixed | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 10 | 0004 | 0.797935 | 0.2084 | 2 | 1 | fixed | value | True | False | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 11 | 0006 | 0.797935 | 0.2084 | 2 | 1 | fixed | downstream_value | True | False | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 12 | 0015 | 0.797935 | 0.2084 | 2 | 1 | fixed | value | False | False | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 13 | 0007 | 0.783369 | 0.2667 | 3 | 1 | adaptive_v6 | value | False | False | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 14 | 0010 | 0.783369 | 0.2667 | 3 | 1 | fixed | downstream_value | True | False | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 15 | 0013 | 0.778080 | 0.2879 | 3 | 0 | fixed | value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 16 | 0009 | 0.769450 | 0.3224 | 4 | 1 | adaptive_v6 | downstream_value | False | False | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 17 | 0001 | 0.761895 | 0.3526 | 4 | 0 | fixed | value | True | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 18 | 0008 | 0.761895 | 0.3526 | 4 | 0 | adaptive_v6 | downstream_value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 19 | 0003 | 0.760276 | 0.3591 | 3 | 1 | fixed | downstream_value | False | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 20 | 0005 | 0.745710 | 0.4173 | 4 | 1 | adaptive_v6 | value | True | True | 0.9000 | 0.8855 | 0.7967 | 0.7200 |

## 6. 分组分析

### 按 topk 分组

| group | n | mean_obj | best_obj | mean_payload | mean_mrr | mean_ndcg | mean_f1 | mean_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 12 | 0.821930 | 0.832511 | 0.1125 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 3 | 4 | 0.776273 | 0.783369 | 0.2951 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 4 | 4 | 0.759737 | 0.769450 | 0.3612 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |

### 按 budget_mode 分组

| group | n | mean_obj | best_obj | mean_payload | mean_mrr | mean_ndcg | mean_f1 | mean_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_v6 | 5 | 0.778587 | 0.832511 | 0.2858 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| fixed | 15 | 0.807618 | 0.832511 | 0.1697 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |

### 按 warmup 分组

| group | n | mean_obj | best_obj | mean_payload | mean_mrr | mean_ndcg | mean_f1 | mean_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 11 | 0.814724 | 0.832511 | 0.1413 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| 1 | 9 | 0.782805 | 0.809265 | 0.2690 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |

### 按 score_mode 分组

| group | n | mean_obj | best_obj | mean_payload | mean_mrr | mean_ndcg | mean_f1 | mean_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| downstream_value | 7 | 0.791135 | 0.832511 | 0.2356 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| value | 13 | 0.805327 | 0.832511 | 0.1789 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |

### 按 layerwise_budget 分组

| group | n | mean_obj | best_obj | mean_payload | mean_mrr | mean_ndcg | mean_f1 | mean_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | 6 | 0.788332 | 0.797935 | 0.2469 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| True | 14 | 0.805515 | 0.832511 | 0.1781 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |

### 按 utility_memory 分组

| group | n | mean_obj | best_obj | mean_payload | mean_mrr | mean_ndcg | mean_f1 | mean_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | 15 | 0.808024 | 0.832511 | 0.1681 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |
| True | 5 | 0.777369 | 0.797935 | 0.2907 | 0.9000 | 0.8855 | 0.7967 | 0.7200 |

## 7. 关键观察

- `topk=2` 是当前搜索中最稳定的高分区域。最佳 objective 达到 `0.832511`，payload 仅约 `0.0701`，明显比 `topk=3/4` 更省通信预算。
- `fixed` budget 并没有输给 `adaptive_v6`。多个最高分 trial 使用 `fixed`，说明在当前 300-example Hotpot proxy 上，复杂 adaptive 扩预算机制没有表现出额外收益。
- `adaptive_v6` 也能达到同等最佳值，例如 `trial_0016`，但其优势并非来自 adaptive 机制本身，而更像是因为最终落入了同样的低预算 `topk=2` 行为模式。
- `warmup=0` 明显更有利。最高分 trial 均为 `warmup=0`，而 `warmup=1` 多数 objective 较低，说明在该设置下额外 warmup 可能没有带来有效泛化收益。
- `utility_memory=False` 在高分区域中更常见。当前 utility memory 没有证明能改善 downstream proxy，可能还会引入滞后或噪声。
- `layerwise_budget=True` 与最高分强相关，说明层级预算约束/分配仍可能是 V6 中较有价值的部分。
- 多个不同参数组合得到完全相同的 downstream metrics，提示当前 300-example proxy 的分辨率仍有限；它能区分 `topk=2/3/4` 这类大结构差异，但对 hard query/client 阈值等细粒度参数不够敏感。

## 8. 科研意义

该实验的价值主要在于把 V6 的复杂机制拆开后进行自动化搜索，结果显示：当前最值得保留和进一步验证的不是“更复杂的 adaptive 扩预算”，而是“低 topk、低 payload、layerwise budget 的稳定配置”。这对论文叙事很重要，因为它提示 V6 后续不应继续盲目堆叠复杂 adaptive 模块，而应转向更严格的预算约束和更高判别力的 hard query evaluation。

换言之，`V6-HP1-OPTUNA` 给出的不是一个“V6 adaptive 完胜”的结论，而是一个更可靠的负/中性结论：复杂策略并未在当前 proxy 下显著超过简洁低预算策略。这可以作为方法迭代依据，帮助后续设计更清晰的 V7 或正式论文实验。

## 9. 局限性

- 搜索规模只有 20 个有效 trial，适合作为小规模 pilot，不足以宣称全局最优。
- 下游评估只使用 300 条 HotpotQA validation proxy，仍可能存在 metric 饱和或样本不足问题。
- 多个 trial 指标完全相同，说明当前 retriever/downstream pipeline 对部分细粒度超参变化不够敏感。
- objective 中 payload penalty 固定为 0.25，不同 penalty 下最佳策略可能变化。
- Trial 0 曾因 launcher 参数解析问题失败，但已修复；最终 20 个有效 trial 均完成，不影响主体结论。

## 10. 后续建议

- 以 `topk=2, warmup=0, layerwise_budget=True, utility_memory=False` 作为后续正式实验候选配置。
- 做同预算复现实验：固定 payload 约 `0.07`，比较 V3/V4/V5/V6/HP1-OPTUNA 最佳配置。
- 扩大 HotpotQA eval examples，例如从 300 增至 1000 或完整 validation，以提高指标判别力。
- 单独设计 hard-query-only subset，避免简单样本拉高所有配置，使 adaptive 机制难以体现价值。
- 重新审视 adaptive_v6：只有当 hard query/client 检测足够准确，并且 downstream proxy 能识别扩预算收益时，adaptive 才有继续强化的意义。
- 若继续 Optuna，建议下一轮缩窄搜索空间，重点比较 `topk=2` 内部的 layerwise、score_mode、payload penalty 和 hard-query subset 表现。

## 11. 文件与产物

| 产物 | 路径 |
| --- | --- |
| Optuna 简要报告 | `V6-HP1-OPTUNA/outputs/optuna_report.md` |
| Optuna JSON 汇总 | `V6-HP1-OPTUNA/outputs/optuna_summary.json` |
| Optuna CSV 汇总 | `V6-HP1-OPTUNA/outputs/optuna_summary.csv` |
| Trial 结果目录 | `V6-HP1-OPTUNA/outputs/trial_results/` |
| RAG eval 输出目录 | `V6-HP1-OPTUNA/outputs/rag_eval/` |
| 运行日志 | `v6_hp1_optuna.nohup.log` |
| 本完整报告 | `V6-HP1-OPTUNA/v6_hp1_optuna_complete_experiment_report_cn.md` |

## 12. 结论

`V6-HP1-OPTUNA` 已完成 20-trial 小规模搜索。实验表明，在当前 HotpotQA proxy 设置下，最佳配置不是更激进的 adaptive 扩预算，而是更克制的低预算配置：`topk=2 + warmup=0 + layerwise_budget=True + utility_memory=False`。该配置以约 `0.0701` 的 payload 取得最高 objective `0.832511`，并达到 `MRR=0.9000, NDCG=0.8855, F1=0.7967, EM=0.7200, Recall@3=0.9133`。

因此，本实验最重要的结论是：V6 后续优化方向应从“增加机制复杂度”转向“低预算稳定配置 + 更高判别力的 hard query 下游评估”。
