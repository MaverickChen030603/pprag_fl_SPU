# V7-agent-BSP-DIAG 当前进展报告

更新时间：2026-06-22 16:30 JST  
远程项目：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## 1. 总体状态

V7-agent-BSP-DIAG 主流程仍在运行，尚未完成最终闭环。相比 2026-06-22 02:18 JST 的状态，official FiD/T5 从 `5/40` 推进到 `6/40`。

当前进程状态：

- 主流程：`PID 97504`，运行约 `3-21:26:46`
- official launcher：`PID 155046`，运行约 `3-11:40:35`
- 当前 official 子进程：`PID 632986`
- 当前 run：`[7/40] agent_bsp_hf_bandit_strict seed=1`
- reader：真实 `FiD/T5 t5-base`
- device：CPU
- passage ordering：`retrieval_score`

硬错误扫描仍为空：未发现 `Traceback`、`CUDA out of memory`、`killed`、`No such file`、`T5Tokenizer requires SentencePiece` 或 `empty predictions`。

## 2. 完成度

| 模块 | 当前完成度 | 说明 |
|---|---:|---|
| training / strict diagnostic | 40/40 | 已完成 |
| official FiD/T5 metrics | 6/40 | 新增完成 `agent_bsp_hf_bandit_strict seed=0` |
| official per-query files | 6/40 | 每个 completed run 有 `per_query_official.jsonl` |
| official predictions | 6/40 | 每个 completed run 有 `hotpot_predictions.json` |
| reader input export | 0 | 仍未开始，需等 official 阶段结束或单独触发 |
| final analysis/report | 未完成 | 仍被 official 40-run 阻塞 |

## 3. 已完成 official 指标

### `agent_bsp_hf_bandit_retrieval` seeds 0-4

5 个 seed 均已完成，mean 指标为：

- answer F1：`0.655278`
- support F1：`0.505600`
- joint F1：`0.331481`
- support title recall：`0.744700`

### `agent_bsp_hf_bandit_strict` seed 0

新增完成的 run：

| method | seed | n | answer F1 | support F1 | joint F1 | support title recall | avg topk | budget std | elapsed seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| agent_bsp_hf_bandit_strict | 0 | 1000 | 0.655958 | 0.506000 | 0.332241 | 0.744500 | 3 | 0.0 | 48129 |

当前只完成了 `agent_bsp_hf_bandit_strict` 的 1 个 seed，不能做稳定 method-level 结论。

## 4. Same-Budget 与 metadata

已完成 6 个 official run 均显示：

- `n = 1000`
- `reader_model = t5-base`
- `beam_size = 3`
- `max_input_length = 768`
- `passage_ordering = retrieval_score`
- `avg_budget_topk = "3"`
- `budget_std = 0.0`

same-budget 在已完成 official metadata 上是对齐的。需要注意：`avg_budget_topk` 仍以字符串 `"3"` 写入，最终 collector/report 应规范化为数值 `3.0`，避免正式表中类型不一致。

## 5. 速度评估

CPU official FiD/T5 仍然非常慢。已完成 6 个 run 的耗时大约在 12.4 到 13.9 小时之间，新增 `agent_bsp_hf_bandit_strict seed=0` 用时 `48129s`，约 13.37 小时。

当前速度下，完整 40-run 仍可能需要数周。此时继续等待可以获得完整矩阵，但会显著拖慢 reader input verification、cache audit 和 final landing report 的闭环。

## 6. Reader Input / Ordering / Cache 风险

reader input export 仍为 `0`，所以以下关键问题还没有被最终验证：

1. `agent_priority` 是否真的改变 FiD/T5 reader input。
2. `gold_oracle_debug` 是否真的把 supporting titles 前置。
3. prediction / reader input cache 是否按 method、seed、ordering 隔离。
4. selector 差异是否传导到 endpoint QA 指标。

此前小样本 smoke 曾显示 ordering hash diff 为 0，因此当前最重要的风险仍是 reader ordering pipeline 没有真正生效。不能用当前 6 个 retrieval_score official 结果解释 agent 方法整体有效或无效。

## 7. 当前判断

可以确认：

1. official FiD/T5 是真实 `t5-base`，不是 fallback reader。
2. 已完成 official runs 的 budget metadata 对齐：Top-K 为 3，budget std 为 0。
3. `agent_bsp_hf_bandit_retrieval` 的 5-seed endpoint 指标非常稳定。
4. `agent_bsp_hf_bandit_strict seed=0` 指标略高于 retrieval 均值，但只是单 seed，不能过度解释。

不能确认：

1. 不能确认 BSP-HF 超过 baseline，因为 baseline official DIAG 对照尚未跑到。
2. 不能确认 agent ordering 有效，因为 reader input verification 尚未生成。
3. 不能确认 endpoint 无提升代表方法无效，因为 ordering/cache/input pipeline 尚未闭环。

## 8. 建议

建议不要只被动等待 40 个 CPU official run 全部串行结束。更稳的推进方式是：

1. 保留当前主流程继续跑，不中断已有日志和 completed outputs。
2. 并行准备一个最小 reader input verification：直接基于已有 6 个 completed official artifacts 与对应 run dirs 导出 reader input hash，先验证 `retrieval_score` 路径是否可复现。
3. 若资源允许，另开 GPU/更快配置跑 method-balanced minimal subset，而不是等完整 40-run。
4. 在最终报告中继续把当前结论限定为“official partial results”，不写 endpoint 胜出结论。

## 9. 当前决策

当前仍不满足论文写作验收。最接近的决策仍是：

`Decision B: 需要先验证或修 reader pipeline`

原因：official 真实结果已经产生，但 reader input verification 尚未开始；在 ordering/hash/cache 闭环前，endpoint QA 指标不能作为 agent 方法成败的最终解释。

