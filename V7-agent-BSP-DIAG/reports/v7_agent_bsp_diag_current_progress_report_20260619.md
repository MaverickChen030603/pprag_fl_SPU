# V7-agent-BSP-DIAG 当前进展报告

更新时间：2026-06-19 14:30 JST 远程快照  
项目路径：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG`

## 1. 实验目的

V7-agent-BSP-DIAG 是在 V7-agent-BSP 平行路径上新建的诊断实验分支，核心目标不是直接宣称 agent 方法成功，而是定位当前 HotpotQA / Federated RAG 流水线中 agent 信号被削弱的环节。

本轮重点诊断三件事：

1. 在 strict same-budget 约束下，`agent_rule_v7` / `agent_bandit_v7` 是否真正保持与 baseline 等价上传预算。
2. official FiD/T5 reader 是否对 selector 差异敏感，避免只看 retrieval 或 fallback 指标造成误判。
3. `gold_oracle_debug`、`retrieval_score`、BSP/hard-feedback/bandit 等不同 evidence ordering 或选择策略，是否真的改变 reader 输入和最终 answer/support 指标。

## 2. 当前运行状态

主流程仍在运行中。

- 主流程 PID：`97504`
- 当前阶段：`scripts/run_v7bspdiag_official_fid.sh`
- 当前 official FiD/T5 run：`agent-bsp-hf-bandit-retrieval seed=0`
- 当前 official 子进程 PID：`155141`
- 主流程命令链：
  - `bash scripts/run_v7bspdiag_suite.sh v7bspdiag_hf`
  - `bash scripts/run_v7bspdiag_official_fid.sh`
  - `python scripts/export_reader_inputs_diag.py ...`
  - `python scripts/analyze_v7bspdiag.py`
  - `python scripts/generate_v7bspdiag_report.py`

## 3. 完成度统计

截至本次查询：

| 模块 | 完成数 | 总数 | 状态 |
|---|---:|---:|---|
| `v7bspdiag_hf` 训练/strict diagnostic | 40 | 40 | 已完成 |
| strict metrics | 40 | 40 | 已生成 |
| final artifacts | 40 | 40 | 已生成 |
| official FiD/T5 metrics | 0 | 40 | 正在第 1 个 run |
| reader input export files | 0 | 待生成 | official 完成后开始 |
| automatic analysis/report | 待执行 | 待执行 | official + export 后执行 |

训练阶段已完成全部 40 个 run，并已处理为 strict diagnostic 结果：

```text
Processed 40 HP1 strict diagnostic runs into .../V7-agent-BSP-DIAG/analysis/strict_runs/v7bspdiag_hf
DONE v7bspdiag_hf
```

随后 official FiD/T5 阶段已启动，环境检查显示 `sentencepiece` 和 `T5Tokenizer.from_pretrained('t5-base')` 可用，因此当前 official 评估是按真实 FiD/T5 路径推进，不是 fallback reader。

## 4. 日志与错误检查

主要日志：

- `/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/runs/v7bspdiag_all.nohup.log`
- `/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/runs/logs/v7bspdiag_hf_20260618_190304.log`
- `/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG/runs/logs/official_fid_diag_20260619_044915.log`

本次扫描未发现硬错误：

- 未发现 `Traceback`
- 未发现 `CUDA out of memory`
- 未发现 `killed`
- 未发现 official tokenizer / sentencepiece 初始化错误

## 5. Same-Budget 状态判断

训练日志显示 post-warmup 阶段的 budget 已稳定在严格 Top-K 附近：

- `avg_budget_topk` 约为 `3.0`
- `avg_payload` 约为 `0.1996`

这说明 BSP-DIAG 分支已经进入严格 same-budget 的预期形态。最终报告仍需要等待 `analyze_v7bspdiag.py` 汇总每个 method/seed 的 `avg_topk`、`budget_std`、payload 差异后，再给出正式预算对齐结论。

当前可给出的中期判断是：训练阶段没有看到隐性预算扩张迹象，但 paper-facing 结论必须以 official/analysis 汇总表为准。

## 6. Reader Sensitivity 并行状态

原 `V7-agent-BSP` 的 reader sensitivity 长跑仍在继续：

- PID：`4100711`
- 已生成 sensitivity metrics：`173`
- 当前 run：`agent-rule-v7-dynamic seed=3`
- 当前配置：`beam1_len768_retrieval_score`
- 当前阶段：CPU FiD evaluation

该 sensitivity 网格非常慢。日志中可见单个 sensitivity run 曾耗时约 `5251.73` 秒，约 87.5 分钟。因此它仍有参考价值，但不应阻塞 BSP-DIAG 主线的 official 40-run 诊断。

## 7. 初步数据解读

### 7.1 训练/strict 阶段

40 个 strict diagnostic run 已完成，说明当前方法组合、预算控制、artifact 生成路径是可跑通的。这个阶段主要证明“方法可以在统一预算和统一 artifact 结构下稳定产出”，还不能证明 agent 优于 baseline。

### 7.2 official FiD/T5 阶段

official 阶段刚进入第 1/40 个 run，因此现在还没有 answer EM/F1、support EM/F1、joint metrics 的 official 对比结论。此时不能宣称 `agent_rule_v7` 或 `agent_bandit_v7` 已经超过 `hypernet_v6` / `adaptive_v6`。

### 7.3 Reader input smoke 发现

此前小样本 reader input smoke 显示：

- `gold_oracle_debug` vs `retrieval_score` 的 `reader_input_hash_diff_rate = 0.0`
- `passage_order_hash_diff_rate = 0.0`

这只是小样本证据，但它提示一个重要风险：如果不同 selector/debug mode 最终送入 FiD/T5 的 passage 集合和顺序没有实质差异，那么 reader 端自然很难拉开 answer/support 指标。BSP-DIAG 后续的 full reader input export 正是为了确认这个问题在 40-run 全量诊断中是否成立。

## 8. 当前结论

截至目前，V7-agent-BSP-DIAG 的工程执行是健康的：

1. 训练与 strict diagnostic 已完整跑完 40/40。
2. official FiD/T5 环境可用，并已开始真实 official eval。
3. 尚未发现硬错误或 OOM。
4. 最终分析还不能开始，因为 official 40-run 与 reader input export 尚未完成。
5. 当前最值得关注的科学问题，是 selector 差异是否真正传递到了 reader 输入层；如果 reader input hash 差异仍接近 0，则 agent 策略即使在上传选择层变化，也可能无法在 official answer/support 指标上体现。

## 9. 下一步

建议按以下顺序继续：

1. 等待 `run_v7bspdiag_official_fid.sh` 完成 40 个 official FiD/T5 run。
2. 自动执行 `export_reader_inputs_diag.py`，检查不同 method/debug mode 的 reader input hash 与 passage order 差异。
3. 执行 `analyze_v7bspdiag.py`，汇总 same-budget、official metrics、reader input sensitivity。
4. 执行 `generate_v7bspdiag_report.py`，生成正式完整报告。
5. 若 reader input 差异仍然不足，下一轮优化应优先修改 evidence ordering / passage construction，而不是继续只调 agent scoring 权重。
