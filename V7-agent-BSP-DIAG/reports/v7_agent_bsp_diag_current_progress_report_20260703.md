# V7-agent-BSP-DIAG 当前进展报告

更新时间：2026-07-03 14:13 JST  
远程项目：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## 1. 总体状态

V7-agent-BSP-DIAG 主流程仍在运行，official FiD/T5 尚未完成 40-run 全量闭环。相比 2026-06-29 的 `18/40`，当前已推进到 `24/40` completed，正在运行第 `25/40`。

当前进程：

- 主流程：`PID 97504`，运行约 `14-19:10:42`
- official launcher：`PID 155046`，运行约 `14-09:24:31`
- 当前 official 子进程：`PID 3999680`
- 当前 run：`[25/40] agent_bsp_memory_bandit_retrieval seed=4`
- reader：真实 `FiD/T5 t5-base`
- device：CPU
- passage ordering：`retrieval_score`

硬错误扫描为空：未发现 `Traceback`、`CUDA out of memory`、`killed`、`No such file`、`T5Tokenizer requires SentencePiece`、fallback 或 empty predictions。

## 2. 完成度

| 模块 | 当前完成度 | 说明 |
|---|---:|---|
| training / strict diagnostic | 40/40 | 已完成 |
| official FiD/T5 metrics | 24/40 | 已完成 24 个真实 FiD/T5 run |
| official per-query files | 24/40 | 每个 completed run 有 `per_query_official.jsonl` |
| official predictions | 24/40 | 每个 completed run 有 `hotpot_predictions.json` |
| reader input export | 0 | 仍未开始 |
| reader ordering verification | 0 | `reader_input_ordering_verification.csv` 尚未生成 |
| analysis CSV | 11 | 主要为此前 strict/sensitivity/partial analysis |
| reports | 8 | 当前报告外已有 8 个 report 文件 |

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

### `agent_bsp_memory_bandit_no_history_state`，5/5 seeds

- answer F1：`0.655398`
- support F1：`0.506000`
- joint F1：`0.331961`
- support title recall：`0.743900`

### `agent_bsp_memory_bandit_retrieval`，4/5 seeds

- answer F1：`0.654558`
- support F1：`0.506000`
- joint F1：`0.331841`
- support title recall：`0.744500`

当前 `agent_bsp_memory_bandit_retrieval seed=4` 正在运行。

## 4. Same-Budget 与 metadata

已完成的 24 个 official run 均显示：

- `n = 1000`
- `reader_model = t5-base`
- `beam_size = 3`
- `max_input_length = 768`
- `passage_ordering = retrieval_score`
- `avg_budget_topk = "3"`
- `budget_std = 0.0`

same-budget 在 completed official metadata 上继续对齐。仍需在最终 collector 中把 `avg_budget_topk` 从字符串 `"3"` 规范化为数值 `3.0`。

## 5. 初步数据解读

当前 endpoint 指标高度接近，方法间差距仍在非常小的范围内：

- 最高的 completed 5-seed 方法是 `agent_bsp_hf_bandit_strict`，joint F1 `0.332161`
- `agent_bsp_memory_bandit_no_history_state` joint F1 `0.331961`
- `agent_bsp_memory_bandit_retrieval` 当前 4-seed joint F1 `0.331841`
- `agent_bsp_hf_bandit_retrieval` joint F1 `0.331481`

这些差距不足以支持“agent endpoint 显著拉开”的表述。当前更像是：strict/agent variants 在真实 FiD/T5 endpoint 下表现稳定，但 QA 指标对 selector 差异并不敏感，至少在 `retrieval_score` ordering 下没有明显分离。

## 6. Reader Input / Ordering / Cache 状态

关键 blocker 仍未解除：

- `debug_reader_inputs` 文件数：0
- `reader_input_ordering_verification.csv`：未生成
- `agent_priority` 是否改变 reader input：未验证
- `gold_oracle_debug` 是否有效前置 gold support：未验证
- prediction / input cache 是否按 ordering 隔离：仍需最终 audit

因此当前 24 个 official runs 只能作为 partial endpoint trace，不能作为最终论文主结论。此前 smoke 曾出现 `reader_input_hash_diff_rate = 0.0`，这个风险仍必须在 full 或 minimal reader verification 中处理。

## 7. 速度评估

CPU FiD/T5 仍然是主要瓶颈。当前完成 24/40，还剩 16 个 run，其中第 25 个正在运行。按近期单 run 约 13-15 小时估计，official 阶段还需要约 8-10 天。

完整 official 后，原主流程才会继续执行：

1. `scripts/export_reader_inputs_diag.py`
2. `scripts/analyze_v7bspdiag.py`
3. `scripts/generate_v7bspdiag_report.py`

## 8. 当前判断

可以确认：

1. official FiD/T5 真实运行，非 fallback。
2. completed official runs 均保持 `topk=3`、`budget_std=0.0`。
3. endpoint 指标稳定，但方法间差距非常小。
4. `agent_bsp_hf_bandit_strict` 当前略高，但没有形成强 endpoint 正信号。

不能确认：

1. 不能确认 BSP-HF 超过全部 baseline，因为 current completed matrix 仍未包含完整 method-balanced baseline/PM/dynamic official 对照。
2. 不能确认 agent ordering 有效，因为 reader input verification 尚未启动。
3. 不能解释 endpoint 差异来源，因为 cache/order/input pipeline 尚未闭环。

## 9. 建议

建议继续保留当前 official 进程，但为了避免继续等待后仍无法解释结果，应并行推进：

1. 对已完成 24 个 run 先做 retrieval_score reader input hash 导出。
2. 最小化重跑或导出 `agent_priority` / `gold_oracle_debug` positive control，确认 ordering 是否真的改变输入。
3. 若 hash diff 仍为 0，优先修 reader construction/cache key，而不是等待 40/40。
4. 最终报告继续采用保守措辞：endpoint partial results 稳定但没有强正信号，reader pipeline verification 是解释成败的必要条件。

## 10. 当前决策

当前仍不满足进入论文写作的最低验收标准。最接近的决策仍是：

`Decision B: 需要先验证或修 reader pipeline`

原因：official 真实结果已经积累到 24/40，但 reader input verification 仍为 0；在 ordering/hash/cache 闭环前，不能把 endpoint QA 指标解释为 agent 方法最终成败。

