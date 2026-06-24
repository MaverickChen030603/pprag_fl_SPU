# V7-agent-BSP-DIAG 当前进展报告

更新时间：2026-06-22 02:18 JST  
远程项目：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## 1. 总体状态

V7-agent-BSP-DIAG 主流程仍在运行，没有完成最终闭环。训练与 strict diagnostic 已完成，official FiD/T5 已从此前的 0/40 推进到 5/40。

当前主流程：

- 主 bash：`PID 97504`，已运行约 `3-07:14:26`
- official launcher：`PID 155046`，已运行约 `2-21:28:15`
- 当前 official 子进程：`PID 552745`
- 当前 run：`[6/40] agent_bsp_hf_bandit_strict seed=0`
- reader：真实 `FiD/T5 t5-base`
- device：CPU
- passage ordering：`retrieval_score`

未发现硬错误：

- no `Traceback`
- no `CUDA out of memory`
- no `killed`
- no `No such file`
- no `T5Tokenizer requires SentencePiece`
- no `empty predictions`

## 2. 完成度

| 模块 | 当前完成度 | 说明 |
|---|---:|---|
| `v7bspdiag_hf` training / strict | 40/40 | 已完成 |
| strict metrics | 40/40 | 已生成 |
| official FiD/T5 metrics | 5/40 | 已完成 `agent_bsp_hf_bandit_retrieval` seeds 0-4 |
| official per-query files | 5/40 | 每个 completed official run 有 `per_query_official.jsonl` |
| official predictions | 5/40 | 每个 completed official run 有 `hotpot_predictions.json` |
| reader input export | 0 | official 40-run 完成后才会自动执行 |
| final analysis/report | 未完成 | 仍被 official 40-run 阻塞 |

## 3. 已完成 official 结果

当前已完成的 5 个 official run 全部属于：

`agent_bsp_hf_bandit_retrieval`

seed-level 指标如下：

| seed | n | answer F1 | support F1 | joint F1 | support title recall | avg topk | budget std | elapsed seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 0.654958 | 0.5055 | 0.331241 | 0.7445 | 3 | 0.0 | 44721 |
| 1 | 1000 | 0.655758 | 0.5050 | 0.331241 | 0.7450 | 3 | 0.0 | 49968 |
| 2 | 1000 | 0.655958 | 0.5055 | 0.331741 | 0.7450 | 3 | 0.0 | 47969 |
| 3 | 1000 | 0.654958 | 0.5065 | 0.331941 | 0.7445 | 3 | 0.0 | 44483 |
| 4 | 1000 | 0.654758 | 0.5055 | 0.331241 | 0.7445 | 3 | 0.0 | 45017 |

seed mean：

- answer F1：`0.655278`
- support F1：`0.505600`
- joint F1：`0.331481`
- support title recall：`0.744700`

注意：当前只完成了一个方法的 5 个 seed，还不能与 `hypernet_v6`、`adaptive_v6`、`agent_rule_v7_dynamic`、`agent_pm_bandit_slot` 做正式 method-balanced 结论。

## 4. Same-Budget 状态

已完成 official metadata 中：

- `avg_budget_topk = 3`
- `budget_std = 0.0`
- `n = 1000`
- `reader_model = t5-base`
- `beam_size = 3`
- `max_input_length = 768`

这说明当前已完成的 5 个 official runs 在 metadata 层满足 same-budget 要求。不过 `avg_budget_topk` 现在写成字符串 `"3"`，后续最终汇总表需要规范化为数值 `3.0`，避免统计脚本或论文表格中出现类型不一致。

## 5. 速度与风险

official FiD/T5 在 CPU 上极慢。已完成 run 的耗时约为：

- 最短：`44483s`，约 12.36 小时
- 最长：`49968s`，约 13.88 小时
- 单 run 平均约 12.9 小时

照这个速度，40 个 run 全部完成约需 21 天左右。当前 5/40 已完成，剩余 35 个 run 若继续 CPU 串行执行，预计还需要约 18-19 天。

因此当前最大工程风险不是崩溃，而是 official full matrix 的 CPU 串行成本过高。

## 6. Reader Input / Cache / Ordering 状态

当前 reader input export 尚未执行：

- `debug_reader_inputs` 文件数：0
- `reader_input_ordering_verification.csv` 仍未生成本轮 DIAG 的新结果
- `cache_reuse_audit.md` 存在，但主要来自之前 reader sensitivity / 旧分析

因此还不能判断：

1. `agent_priority` 是否真的改变 FiD/T5 reader input。
2. `gold_oracle_debug` 是否真的把 gold supporting title 前置。
3. official prediction 是否存在 cache / ordering 复用污染。
4. selector 差异是否传递到了 endpoint QA。

此前小样本 smoke 的 `reader_input_hash_diff_rate = 0.0` 仍然是关键风险，但还不是最终结论。

## 7. 当前科学判断

当前可以说：

1. `agent_bsp_hf_bandit_retrieval` 在 strict diagnostic 中保持了 strict same-budget，并且 early evidence recall 已不再是 0。
2. `agent_bsp_hf_bandit_retrieval` 的 completed official seed 指标非常稳定，joint F1 约 `0.3315`。
3. official FiD/T5 确实在真实 `t5-base` reader 上运行，不是 fallback。

当前不能说：

1. 不能说 BSP-HF 已经超过 baseline，因为 baseline official DIAG runs 尚未完成。
2. 不能说 agent ordering 有效，因为 reader input verification 尚未完成。
3. 不能说 endpoint 无提升就是方法无效，因为 ordering/cache/input pipeline 仍未闭环。

## 8. 当前建议

建议不要继续让 40 个 official CPU run 串行跑 18-19 天而不干预。更合理的下一步是：

1. 保留当前进程和日志，不立即杀。
2. 单独做一个最小迁移/加速方案：将后续 official eval 迁移到可用 GPU 或降低并行矩阵规模，先保证 method-balanced 核心对照完成。
3. 优先跑最小闭环 subset：`agent_bsp_hf_bandit_retrieval`、`agent_pm_bandit_slot`、`agent_rule_v7_dynamic`、`hypernet_v6`，每个至少 seed 0-4 或先 seed 0-2。
4. 在继续扩 official 前，先独立执行 reader input export / ordering verification，确认 `gold_oracle_debug` 和 `agent_priority` 是否真的改变输入。
5. 若 ordering hash 仍为 0，优先修 reader input construction / cache key，而不是继续等 40-run endpoint。

## 9. 当前决策

当前不满足进入论文写作的最低验收标准。最接近的判断是：

`Decision B: 需要先验证或修 reader pipeline`

原因：official 已经开始产出真实 FiD/T5 结果，但 reader input verification 仍未完成，而此前 smoke 曾显示不同 ordering 的 reader input hash 完全相同。最终结论必须等 ordering verification 和 cache audit 完成后才能写。

