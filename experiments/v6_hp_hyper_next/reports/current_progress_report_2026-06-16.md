# V6-HP-hyper 下一阶段实验进展报告

- 报告时间：2026-06-16 14:27 JST
- 本地仓库：`/Users/iilab/PPRAG_FL/FedE4RAG-main/FedE4RAG-main`
- 服务器仓库：`/home/iiserver31/projects/FedE4RAG-main`
- 新阶段实验目录：`experiments/v6_hp_hyper_next/`
- 当前状态：实验框架已建立，Task A 的 HotpotQA per-query 输入数据已定位到数据加载错误并完成修复，正在服务器重跑修复后的 `all1500_v3` per-query 生成。

## 1. 实验目的

本阶段实验的核心目的不是继续堆叠新的 FedRAG 机制，而是解决当前 V6-HP-hyper / hypernet_v6 实验中的评测瓶颈，并为论文级结论建立更严格的证据链。

前一阶段 V6-HP1 与 V6-HP1-OPTUNA 已经显示，当前最稳定的低通信配置为：

| 参数 | 当前 anchor |
| --- | --- |
| `topk` | 2 |
| `warmup` | 0 |
| `budget_mode` | fixed |
| `score_mode` | value |
| `use_utility_memory` | False |
| `layerwise_budget` | True |
| Payload | 约 0.070134 |

该配置在现有 HotpotQA proxy 上达到约：

| Metric | Value |
| --- | ---: |
| MRR | 0.9000 |
| NDCG | 0.8855 |
| F1 | 0.7967 |
| EM | 0.7200 |
| Recall@3 | 0.9133 |

但这些结果还不足以支撑“adaptive budget 本身已经优于 fixed low-budget”的结论。原因是当前 evaluation proxy 存在指标饱和，多个 trial 得到完全相同的 MRR/F1/EM，且此前 hard-query-only subset 只有约 50 条，远低于论文实验需要的 500-1000 条。

因此，本阶段实验要回答五个研究问题：

1. 在严格相同 payload 约束下，V6-HP-hyper 是否优于 V3/V4/V5？
2. V6 的优势是否主要体现在 hard query subset 上？
3. `layerwise_budget` 是否是低通信预算下稳定性的关键因素？
4. adaptive budget 在总 payload 不增加时是否真的优于 fixed low-budget？
5. utility memory 当前是有效机制，还是应降级为 short-term/tie-breaker 辅助机制？

## 2. 当前实验配置

新阶段配置已集中记录在：

```text
experiments/v6_hp_hyper_next/configs/all_used_configs.yaml
```

核心配置如下：

| 项目 | 设置 |
| --- | --- |
| 实验名称 | `v6_hp_hyper_next` |
| 目标 payload | 0.070134 |
| payload tolerance | ±0.002 |
| seeds | 42, 43, 44 |
| evaluation splits | `all_1000`, `hard_500`, `hard_1000`, `full_validation` |
| 主线 anchor | V6-HP-hyper best |
| 主线策略 | `topk=2`, `warmup=0`, `fixed`, `score_mode=value`, `utility_memory=False`, `layerwise_budget=True` |

新阶段输出目录已经创建：

```text
experiments/v6_hp_hyper_next/
├── configs/
├── figures/
├── logs/
├── reports/
├── results/
└── subsets/
```

服务器当前 GPU 状态检查结果显示，A100 GPU 基本空闲，适合继续执行修复后的 HotpotQA per-query 生成与后续同预算 benchmark。

## 3. 方法设计

### 3.1 Hard Query Benchmark 重构

此前 hard subset 主要依赖“完全 miss”或过严筛选，导致只得到约 50 条 hard query，样本量不足。因此本阶段已将 hard query 构建逻辑改为 difficulty ranking，而不是只保留完全 miss。

新的 difficulty score 设计为：

```text
difficulty_score =
0.35 * gold_rank_score
+ 0.25 * (1 - recall@3)
+ 0.20 * (1 - answer_coverage)
+ 0.10 * (1 - support_doc_coverage)
+ 0.10 * (1 - bridge_entity_coverage)
```

脚本位置：

```text
experiments/v6_hp_hyper_next/build_ranked_hotpot_subsets.py
```

设计上会输出：

```text
hotpot_easy_1000.json
hotpot_medium_1000.json
hotpot_hard_500.json
hotpot_hard_1000.json
hotpot_all_1000.json
hotpot_full_eval.json
```

并生成：

```text
experiments/v6_hp_hyper_next/reports/hard_subset_stats.md
experiments/v6_hp_hyper_next/results/hard_subset_stats.csv
```

### 3.2 Same-Payload Baseline Benchmark

后续核心 benchmark 将以 payload 约 `0.070134 ± 0.002` 为硬约束，比较：

| 方法 | 角色 |
| --- | --- |
| V3 | 早期 hypernet selective-upload baseline |
| V4 | 强异构鲁棒性改进 baseline |
| V5 | 同预算与预算膨胀控制 baseline |
| V6-HP-hyper best | 当前低预算 anchor |

该实验必须避免用高 payload baseline 与低 payload V6 直接比较。若某个版本无法自然达到目标 payload，需要通过 topk 或 budget ratio 校准，并记录实际 payload。

### 3.3 V6 Ablation

消融实验将围绕当前 anchor，而不是重新大范围搜索。重点比较：

| 配置 | 目的 |
| --- | --- |
| `v6_fixed_anchor` | 当前最稳定低预算配置 |
| `v6_no_layerwise` | 验证 layerwise budget 是否关键 |
| `v6_delta_score` | 验证 score mode 替换是否影响低预算选择 |
| `v6_utility_memory_full` | 检查完整 utility memory 是否引入滞后/噪声 |
| `v6_utility_memory_ema` | 检查短期 EMA memory 是否更稳 |
| `v6_hard_weighting_off` | 移除 hard query weighting 的对照 |
| `v6_hard_weighting_on` | 保留 hard query weighting 的对照 |

### 3.4 Adaptive Same-Payload Verification

adaptive budget 的验证将严格限制总通信预算，不允许通过整体 payload 增加获得不公平收益。

后续要比较：

| 配置 | 约束 |
| --- | --- |
| `fixed_anchor` | 固定低预算基线 |
| `adaptive_realloc_same_payload` | hard client/query 得到更多局部预算，但从 easy client/layer 挪预算，总 payload 不增加 |
| `adaptive_capped_same_payload` | 局部 adaptive 后进行 round-level cap/rescale |
| `adaptive_v6_original` | 原始 adaptive 对照，主要用于观察是否存在 payload 膨胀 |

判断标准将保持保守：只有 adaptive 在 `hard_1000` 上 MRR 提升至少 0.02、F1 提升至少 0.015，并且 payload 不超过 fixed anchor，才认为 adaptive 具有独立潜在价值。

## 4. 已完成工作

### 4.1 旧阶段实验与报告

已经完成：

| 阶段 | 状态 | 主要结论 |
| --- | --- | --- |
| V3 | 已完成 | 建立基础 hypernet selective-upload 框架 |
| V4 | 已完成 | 强异构下鲁棒性提升，但存在预算膨胀问题 |
| V5 | 已完成 | 改进预算控制，但下游指标仍难拉开差距 |
| V6-HP1 | 已完成 | 引入 HotpotQA，提高任务难度 |
| V6-HP1-OPTUNA 第一轮 | 已完成 | 找到低预算 anchor，`topk=2` 区域最稳定 |
| V6-HP1-OPTUNA Stage2 | 已完成 | 窄搜索继续支持低预算 fixed anchor，但 hard subset 仍不足 |

### 4.2 V6-HP1-OPTUNA 已确认结果

第一轮 Optuna 完成 20 个有效 trial，最佳结果：

| 项目 | 结果 |
| --- | ---: |
| Best trial | trial_0002 |
| Best objective | 0.832511 |
| Payload | 0.070134 |
| MRR | 0.9000 |
| NDCG | 0.8855 |
| F1 | 0.7967 |
| EM | 0.7200 |
| Recall@3 | 0.9133 |

Stage2 完成 24 个 trial，最佳结果：

| 项目 | 结果 |
| --- | ---: |
| Best trial | trial_0010 |
| Best objective | 0.843032 |
| Payload | 0.070134 |
| Best payload penalty | 0.1 |
| MRR | 0.9000 |
| NDCG | 0.8855 |
| F1 | 0.7967 |
| EM | 0.7200 |
| Recall@3 | 0.9133 |

两轮结果共同支持：当前最可靠的主线是 low-budget fixed anchor，而不是直接宣称 adaptive 已经取得决定性优势。

### 4.3 新阶段目录与脚本

已完成：

```text
experiments/v6_hp_hyper_next/
experiments/v6_hp_hyper_next/configs/all_used_configs.yaml
experiments/v6_hp_hyper_next/build_ranked_hotpot_subsets.py
experiments/v6_hp_hyper_next/logs/all_commands.log
```

已确认存在或已使用的相关脚本：

```text
run_v6_hp1_all.sh
run_v6_hp1_optuna.sh
run_v6_hp1_optuna_stage2.sh
run_v6_hp1_stage2_full.sh
run_same_payload_baseline_benchmark.sh
V6-HP1/build_hotpot_hard_subset.py
V6-HP1-OPTUNA/v6_hp1_optuna.py
V6-HP1/run_rag_eval.py
```

## 5. 当前发现的问题与修复

### 5.1 关键问题：服务器评估实际仍在读取旧金融 JSON 数据

在重建 `v6_hp_hyper_next` hard query benchmark 时，服务器上两次生成的 per-query 文件都只有 50 行：

```text
experiments/v6_hp_hyper_next/results/baseline_all1500_per_query.jsonl
experiments/v6_hp_hyper_next/results/baseline_all1500_v2_per_query.jsonl
```

对应 hard subset stats 显示：

| subset | num_examples |
| --- | ---: |
| hotpot_easy_1000 | 50 |
| hotpot_medium_1000 | 50 |
| hotpot_hard_500 | 50 |
| hotpot_hard_1000 | 50 |
| hotpot_all_1000 | 50 |
| hotpot_full_eval | 50 |

日志进一步显示，虽然 `run_rag_eval.py` 已传入：

```text
RAGTEST_DATASET=hotpot_qa
HOTPOT_SPLIT=validation
HOTPOT_MAX_EXAMPLES=1500
RAGTEST_N=1500
```

但实际 stdout 中仍出现 Pepsico、Coca Cola、General Mills 等金融问答样本，说明评估没有真正切换到 HotpotQA。

### 5.2 根因

服务器上的 `RAGTest/config.py` 版本缺少环境变量 override，导致：

```text
Config().dataset = json_download
```

因此，即使外层命令设置了 `RAGTEST_DATASET=hotpot_qa`，RAGTest 内部仍按 `config.toml` 读取旧的 `json_download`，最终只生成 50 条金融数据 per-query。

这也解释了此前 hard subset 只有 50 条、指标仍饱和、subset JSON 文件虽然存在但实际样本量不足的问题。

### 5.3 已完成修复

已将本地正确版本同步到服务器：

```text
RAGTest/config.py
```

修复后 smoke test 结果：

```text
dataset= hotpot_qa
n_attr= 5
qa_loader_file= /home/iiserver31/projects/FedE4RAG-main/RAGTest/data/qa_loader.py
hotpot_split: validation
hotpot_examples: 5
```

前两条问题已经变为 HotpotQA validation 问题：

```text
What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
```

这说明数据集切换问题已修复。

### 5.4 当前正在运行

修复后已在服务器启动新的 per-query 生成：

```text
baseline_all1500_v3
PID=3895223
```

输出路径：

```text
experiments/v6_hp_hyper_next/results/baseline_all1500_v3_per_query.jsonl
experiments/v6_hp_hyper_next/results/baseline_all1500_v3_rag_eval/
experiments/v6_hp_hyper_next/logs/build_baseline_all1500_v3.nohup.log
```

该任务完成后，需要用新的 `baseline_all1500_v3_per_query.jsonl` 重新生成 difficulty-ranked subsets。旧的 50 条 stats 只能作为失败排查记录，不应作为正式 hard benchmark 结果使用。

## 6. 当前实验数据分析

### 6.1 已确认结论

当前可以确认的结论包括：

1. V6-HP1-OPTUNA 在小规模搜索中找到稳定低预算配置，payload 约 0.070134。
2. `topk=2`、`warmup=0`、`layerwise_budget=True`、`use_utility_memory=False` 是目前最稳定的组合。
3. `fixed` low-budget 并没有输给 `adaptive_v6`，因此 adaptive 的独立优势尚未成立。
4. `utility_memory=True` 未显示稳定收益，可能引入滞后或噪声。
5. `layerwise_budget=True` 与高分 trial 更一致，是目前最值得保留的 V6 机制之一。

### 6.2 暂不能支持的结论

当前还不能支持：

1. 不能宣称 adaptive budget 已经在同等 payload 下显著优于 fixed anchor。
2. 不能用旧的 50 条 hard subset 证明 hard-query setting 下 V6 优势更明显。
3. 不能用当前已生成的 `hard_subset_stats.md` 作为正式论文表格，因为其输入实际来自旧金融数据而非 HotpotQA。
4. 不能直接比较 V3/V4/V5/V6，除非完成 payload≈0.070134 的同预算校准。

### 6.3 当前最重要的实验判断

目前最强、最稳妥的判断是：

```text
在当前已完成实验中，低预算选择性上传能够在 payload≈0.0701 下保持较好的 HotpotQA proxy 检索表现；但 adaptive budget 在严格同等 payload 下的独立优势尚未被验证。
```

## 7. 后续 Todo List

### 7.1 立即执行

1. 等待 `baseline_all1500_v3` per-query 生成完成。
2. 检查新 per-query 文件是否达到接近 1500 行，且问题文本确认为 HotpotQA。
3. 使用 `build_ranked_hotpot_subsets.py` 重新生成：

```text
hotpot_easy_1000.json
hotpot_medium_1000.json
hotpot_hard_500.json
hotpot_hard_1000.json
hotpot_all_1000.json
hotpot_full_eval.json
```

4. 重新生成正式版：

```text
hard_subset_stats.md
hard_subset_stats.csv
```

### 7.2 Task B：Same-Payload Baseline

1. 确认 V3/V4/V5/V6 各自可达到 payload≈0.070134±0.002。
2. 对每个方法运行 seeds 42/43/44。
3. 在 `all_1000`、`hard_500`、`hard_1000` 上评估。
4. 输出：

```text
same_payload_baseline_raw.csv
same_payload_baseline_summary.csv
same_payload_baseline_report.md
```

### 7.3 Task C：V6 Ablation

1. 跑 `v6_fixed_anchor`。
2. 跑 `v6_no_layerwise`。
3. 跑 `v6_delta_score`。
4. 跑 `v6_utility_memory_full` 与 EMA 版本。
5. 跑 `v6_hard_weighting_off/on`。
6. 输出 raw/summary/report。

### 7.4 Task D：Adaptive Same-Payload

1. 实现或检查 `adaptive_realloc_same_payload`。
2. 实现或检查 `adaptive_capped_same_payload`。
3. 与 `fixed_anchor` 和原始 `adaptive_v6` 对比。
4. 重点观察 hard_1000 上 MRR/F1 是否在同 payload 下真正提升。

### 7.5 Final Report

完成后输出：

```text
experiments/v6_hp_hyper_next/reports/final_experiment_report.md
```

并明确区分：

```text
Confirmed findings
Potential findings
Not supported yet
```

## 8. 风险与注意事项

1. 旧的 `baseline_all1500` 与 `baseline_all1500_v2` 结果不能作为正式 HotpotQA 结果使用。
2. 新的 HotpotQA 评估必须先确认 per-query 数量与问题文本，否则可能再次被缓存或配置污染。
3. 如果 `all1500_v3` 运行正常但耗时较长，可以先构建 `hard_500` 和 `all_1000`，再扩展到 full validation。
4. 服务器 `/home` 使用率较高，后续大量 full validation 输出需要注意磁盘空间。
5. Adaptive 结果必须严格记录实际 payload，不能只看 MRR/F1。

## 9. 当前结论

当前阶段已经完成新实验框架搭建和关键数据加载 bug 修复。此前 hard subset 只有 50 条的主要原因不是 HotpotQA 本身 hard query 不足，而是服务器 RAGTest 配置没有应用 `RAGTEST_DATASET=hotpot_qa`，实际仍在读取旧的 `json_download` 金融数据。

修复后，服务器 smoke test 已确认 RAGTest 能正确读取 HotpotQA validation。新的 `baseline_all1500_v3` per-query 生成已启动，完成后即可重新构建正式 hard query benchmark，并继续推进 same-payload baseline、V6 ablation 和 adaptive same-payload 验证。

Under strict same-payload constraints, the strongest supported claim is: low-budget selective upload with layerwise budget can preserve FedRAG retrieval performance at payload≈0.0701, while adaptive budget is not yet independently validated under the current evaluation setting.
