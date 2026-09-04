# V7-agent-BSP-DIAG 当前进展报告

更新时间：2026-06-29 13:53 JST  
远程项目：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## 1. 总体状态

V7-agent-BSP-DIAG 主流程仍在运行，official FiD/T5 尚未完成全量闭环。相比 2026-06-22 16:30 JST 的 `6/40`，当前已推进到 `18/40` completed，正在运行第 `19/40`。

当前进程：

- 主流程：`PID 97504`，运行约 `10-18:50:16`
- official launcher：`PID 155046`，运行约 `10-09:04:05`
- 当前 official 子进程：`PID 2332637`
- 当前 run：`[19/40] agent_bsp_memory_bandit_no_history_state seed=3`
- reader：真实 `FiD/T5 t5-base`
- device：CPU
- passage ordering：`retrieval_score`

硬错误扫描为空：未发现 `Traceback`、`CUDA out of memory`、`killed`、`No such file`、`T5Tokenizer requires SentencePiece`、fallback 或 empty predictions。

## 2. 完成度

| 模块 | 当前完成度 | 说明 |
|---|---:|---|
| training / strict diagnostic | 40/40 | 已完成 |
| official FiD/T5 metrics | 18/40 | 已完成 18 个真实 FiD/T5 run |
| official per-query files | 18/40 | 每个 completed run 有 `per_query_official.jsonl` |
| official predictions | 18/40 | 每个 completed run 有 `hotpot_predictions.json` |
| reader input export | 0 | 仍未开始，需等 official 阶段结束或单独触发 |
| reader ordering verification | 0 | `reader_input_ordering_verification.csv` 尚未生成 |
| analysis CSV | 11 | 主要为此前 strict/sensitivity/partial analysis |
| reports | 7 | 当前报告外已有 7 个 report 文件 |

## 3. 已完成 official 方法

### `agent_bsp_hf_bandit_retrieval`，5/5 seeds

- answer F1：`0.655278`
- support F1：`0.505600`
- joint F1：`0.331481`
- support title recall：`0.744700`

### `agent_bsp_hf_bandit_strict`，5/5 seeds

- answer F1：`0.655878`
- support F1：`0.506100`
- joint F1：`0.332161`
- support title recall：`0.744200`

### `agent_bsp_memory_bandit_no_failure_state`，5/5 seeds

- answer F1：`0.654851`
- support F1：`0.505900`
- joint F1：`0.331708`
- support title recall：`0.744500`

### `agent_bsp_memory_bandit_no_history_state`，3/5 seeds

- answer F1：`0.655824`
- support F1：`0.506000`
- joint F1：`0.332174`
- support title recall：`0.743833`

当前 `no_history_state` seed 3 正在运行，seed 4 尚未开始。

## 4. Same-Budget 与 metadata

已完成的 18 个 official run 均显示：

- `n = 1000`
- `reader_model = t5-base`
- `beam_size = 3`
- `max_input_length = 768`
- `passage_ordering = retrieval_score`
- `avg_budget_topk = "3"`
- `budget_std = 0.0`

same-budget 在已完成 official metadata 上保持对齐。仍需在最终 collector 中把 `avg_budget_topk` 从字符串 `"3"` 规范化为数值 `3.0`。

## 5. 初步数据解读

当前 endpoint 指标非常接近，差距主要在第三位小数内：

- `agent_bsp_hf_bandit_strict` 当前 mean joint F1 为 `0.332161`
- `agent_bsp_memory_bandit_no_history_state` 当前 3-seed mean joint F1 为 `0.332174`
- `agent_bsp_hf_bandit_retrieval` mean joint F1 为 `0.331481`
- `agent_bsp_memory_bandit_no_failure_state` mean joint F1 为 `0.331708`

这说明在当前 retrieval_score ordering 下，BSP-HF / ablation 之间 endpoint QA 差异很小。不能据此宣称 agent 明显拉开 baseline，也不能宣称方法无效；关键原因是 reader input ordering / cache / positive control 尚未闭环。

## 6. Reader Input / Ordering / Cache 状态

当前最重要的 blocker 仍是 reader input verification：

- `debug_reader_inputs` 文件数：0
- `reader_input_ordering_verification.csv`：未生成
- `gold_oracle_debug` 在本轮 DIAG official 中尚未完成验证
- `agent_priority` 是否改变 reader input 尚未验证

因此当前 official 结果只能作为 partial endpoint trace，不能作为最终论文结论。此前 smoke 曾出现 `reader_input_hash_diff_rate = 0.0`，这个风险仍未解除。

## 7. 速度评估

CPU FiD/T5 仍然是主要瓶颈。每个 run 大约 12.5 到 15.7 小时。当前完成 18/40，还剩 22 个 run，其中第 19 个正在运行。

如果继续 CPU 串行，剩余 official 阶段大约还需要 11-14 天。完整 official 后才会自动触发 reader input export、analysis 和 report generation。

## 8. 当前判断

可以确认：

1. official FiD/T5 真实运行，非 fallback。
2. completed official runs 均保持 `topk=3`、`budget_std=0.0`。
3. 当前 endpoint 指标稳定，但方法间差距非常小。
4. `agent_bsp_hf_bandit_strict` 与 `no_history_state` 暂时略高于 `retrieval`，但幅度不足以直接写 endpoint 胜出。

不能确认：

1. 不能确认 BSP-HF 超过 baseline，因为 `hypernet_v6` / `adaptive_v6` / PM / dynamic 等 method-balanced 对照尚未完整进入 current official matrix。
2. 不能确认 agent ordering 有效，因为 reader input verification 尚未启动。
3. 不能解释 endpoint 差异的原因，因为 cache/order/input pipeline 尚未审完。

## 9. 建议

建议继续保留当前 official 进程，但不要只被动等待：

1. 并行准备最小 reader input export，对已完成 18 个 run 先导出 retrieval_score input hash。
2. 单独跑 `agent_priority` / `gold_oracle_debug` 的小样本 positive control，验证 ordering 是否真的改变 FiD/T5 输入。
3. 若 hash diff 仍接近 0，优先修 reader construction/cache key，而不是等待 40/40。
4. 最终报告应继续采用保守措辞：当前 endpoint partial results 没有强正信号，必须先完成 reader pipeline verification。

## 10. 当前决策

当前仍不满足进入论文写作的最低验收标准。最接近的决策仍是：

`Decision B: 需要先验证或修 reader pipeline`

原因：official 真实结果已经积累到 18/40，但 reader input verification 仍为 0；在 ordering/hash/cache 闭环前，不能把 endpoint QA 指标解释为 agent 方法最终成败。

