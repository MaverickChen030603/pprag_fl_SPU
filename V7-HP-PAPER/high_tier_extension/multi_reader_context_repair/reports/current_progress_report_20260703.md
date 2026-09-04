# V7-HP-PAPER 当前进展报告

生成时间：2026-07-03 14:23 JST

## 1. 总体状态

当前 `V7-HP-PAPER` 主线仍保持冻结状态：HotpotQA v2.3 主结果未被修改，未启动 v2.4，未启动 2Wiki1000，未启动 MuSiQue，全量论文材料包和 high-tier extension 材料均已生成。

本轮最新进展集中在：

`V7-HP-PAPER/high_tier_extension/multi_reader_context_repair/`

目标是修复上一轮 multi-reader replication 的 blocker：缺少 final_1000 baseline / selected context text。

## 2. Context Materialization 结果

Context 修复已成功完成。系统从：

`/home/iiserver31/projects/FedE4RAG-main/V7-HP4/data/hotpot_validation_1000.json`

恢复了 v2.3 final_1000 对应的 question、answer、reference context，并结合 v2.3 frozen selected actions 生成：

`outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl`

审计结果：

| 指标 | 数值 |
| --- | --- |
| num_examples | 1000 |
| num_with_question | 1000 |
| num_with_answer | 1000 |
| num_with_baseline_context | 1000 |
| num_with_selected_context | 1000 |
| num_with_support_labels | 1000 |
| avg_baseline_docs | 4.986 |
| avg_selected_docs | 4.986 |
| num_missing_context | 0 |
| num_missing_answer | 0 |
| num_missing_support | 0 |

`context_snapshot_audit.json` 状态为 `pass`，满足 reader replication 的 context 前置条件。

## 3. Multi-Reader Replication 状态

已尝试运行额外 reader：

`google/flan-t5-base`

结果：未完成，状态为 `failed`。

失败原因不是 context 问题，而是服务器环境问题：

- 本地 HuggingFace cache 中没有 `google/flan-t5-base`；
- 服务器当前 Transformers/HF hub 出站查找被禁用；
- 因此 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 无法加载模型；
- 所有 GPU 当前也被高占用，不适合抢占。

reader summary 已写入：

`outputs/reader_outputs/google__flan-t5-base/reader_run_summary.json`

当前 multi-reader 表：

| reader | answer_f1_delta | joint_f1_delta | support_recall@5_delta | sp_f1_delta | answer_f1_p | joint_f1_p | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| google/flan-t5-large_frozen_main | +0.0023 | +0.0150 | +0.0190 | +0.0254 | 0.3625 | 0.0245 | frozen main reader; not rerun |
| google/flan-t5-base | NA | NA | NA | NA | NA | NA | failed |

## 4. 服务器运行状态

当前没有 `multi_reader_context_repair` 的活跃 reader 进程。

服务器上仍有其他项目进程在跑，主要包括：

- `V7-agent-BSP` CPU official/FID evaluation，约运行 25 分钟；
- `V7-agent-BSP-DIAG` CPU official/FID evaluation，约运行 16 小时；
- `V6-HP-hyper` / `RAGTest` hard1000 seed43 RAG eval，约运行 3 分钟；
- 另有非本项目 T5/Gemma 评估进程长期驻留。

GPU 当前占用较高：

| GPU | Used / Total | Util |
| --- | --- | --- |
| 0 | 36199 / 40960 MiB | 41% |
| 1 | 35214 / 40960 MiB | 51% |
| 2 | 29620 / 40960 MiB | 44% |
| 3 | 40261 / 40960 MiB | 52% |

因此目前不建议强行启动 flan-t5-base / xl reader。

## 5. 已完成材料

服务器主目录：

`/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/high_tier_extension/multi_reader_context_repair/`

关键产物：

- `outputs/audit/context_source_inventory.json`
- `outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl`
- `outputs/context_snapshots/context_snapshot_summary.json`
- `outputs/audit/context_snapshot_audit.json`
- `outputs/metrics/multi_reader_metrics.json`
- `outputs/metrics/multi_reader_significance.json`
- `outputs/tables/multi_reader_replication_table.md`
- `reports/context_source_inventory.md`
- `reports/context_materialization_report.md`
- `reports/multi_reader_replication_report.md`
- `reports/multi_reader_claim_boundary.md`

本地归档目录：

`/Users/iilab/ForAgent/实验分析报告/V7-HP-PAPER/high_tier_extension/multi_reader_context_repair/`

## 6. 论文结论边界

可以写：

- frozen v2.3 的 final_1000 context snapshots 已成功 materialize；
- multi-reader replication 已具备 context 条件；
- 当前额外 reader 未完成，原因是模型缓存/服务器环境限制；
- multi-reader 仍应作为 appendix / robustness attempt / limitation。

不能写：

- multi-reader robustness 已验证；
- v2.3 universally improves all readers；
- answer_f1 across readers 显著提升；
- 该结果提升或覆盖 HotpotQA v2.3 主结果。

## 7. 当前建议

短期论文处理：

- 保留 HotpotQA v2.3 为主结果；
- 将 context materialization 成功作为 appendix 可复现实验准备；
- 将 flan-t5-base 未完成作为 honest limitation；
- 投稿目标仍建议 Findings / COLING 更稳，除非后续补齐 reader 模型缓存并完成 flan-t5-base 或 flan-t5-xl 复评。

下一步若要继续补强：

1. 在服务器本地缓存 `google/flan-t5-base` 或提供可用本地模型路径；
2. 等 GPU 释放，或明确使用 CPU 长时间运行；
3. 重新执行：

```bash
cd /home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/high_tier_extension/multi_reader_context_repair
/home/iiserver31/anaconda3/bin/python 03_run_reader_replication.py
/home/iiserver31/anaconda3/bin/python 04_evaluate_multi_reader_outputs.py
/home/iiserver31/anaconda3/bin/python 05_write_multi_reader_report.py
```

