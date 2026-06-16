# V7-agent-2 目前实验进展报告

生成日期：2026-06-16  
项目路径：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2`  
本地报告路径：`/Users/iilab/ForAgent/实验分析报告/V7-agent-2/v7_agent2_experiment_progress_report_20260616.md`

## 1. 当前状态概览

V7-agent-2 已完成本轮核心实验闭环：ablation、bandit early reward、dynamic early-slot 三类实验均已跑完，strict diagnostic 已完成汇总；已完成 29 个 run 的 official-style Hotpot eval，但由于服务器环境缺少 `sentencepiece`，`t5-base` FiD reader 未能加载，实际退回为 extractive fallback reader。因此，当前结论应分成两层：

1. strict diagnostic 层面：agent 机制有清晰正信号，尤其是 early prior、coverage replacement、dynamic early-slot。
2. official eval 层面：当前 fallback reader 结果没有形成显著方法差异，不能作为“FiD/T5 正式结果”使用。

代码与报告已同步 GitHub，最新相关提交为：

- `e55063c Refresh V7-agent-2 official eval summary`
- `9d878f4 Add V7-agent-2 dynamic strict summary`
- `4c9363a Update V7-agent-2 official eval analysis`

## 2. 实验完成情况

| 实验模块 | 完成状态 | run 数 | 备注 |
|---|---:|---:|---|
| `v7agent2_ablation` | 已完成 | 20/20 | 4 方法 × 5 seeds |
| `v7agent2_bandit` | 已完成 | 9/9 | 3 early reward 权重 × 3 seeds |
| `v7agent2_dynamic` | 已完成 | 5/5 | dynamic early-slot × 5 seeds |
| HP1 strict diagnostic | 已完成 | 34 runs | ablation + bandit + dynamic 均已产出 |
| official-style eval | 已完成 | 29/29 | ablation + bandit；fallback reader |

dynamic 曾因 GPU1 OOM 中断，后改用 GPU0 单独重启完成。最终无 OOM/Traceback 残留，GPU 已释放。

## 3. Strict Diagnostic 核心结果

### 3.1 Ablation 结果

| 方法 | n | early recall | HP1 score | bridge recall | avg top-k |
|---|---:|---:|---:|---:|---:|
| `agent_rule_v7` | 5 | 0.2000 | 0.5034 | 0.5985 | 3.0 |
| `agent_rule_v7_no_prior` | 5 | 0.0000 | 0.4234 | 0.6036 | 3.0 |
| `agent_rule_v7_no_coverage` | 5 | 0.0000 | 0.4476 | 0.6349 | 3.0 |
| `agent_rule_v7_no_memory` | 5 | 0.2000 | 0.5034 | 0.5985 | 3.0 |

解释：

- 去掉 early prior 后，early evidence recall 从 0.20 降到 0，HP1 score 从 0.5034 降到 0.4234。
- 去掉 coverage replacement 后，early evidence recall 同样归零，HP1 score 降到 0.4476。
- 去掉 memory 后，本轮结果与 full `agent_rule_v7` 基本一致，说明当前设置下 memory 还没有形成独立增益。
- Wilcoxon 检验中 full vs no-prior、full vs no-coverage 的 p 值为 0.0625，方向一致但未达到 0.05 显著性。样本量只有 5 seeds，统计功效有限。

### 3.2 Dynamic Early-slot 结果

| 方法 | n | avg top-k | early recall | bridge recall | target recall | diversity | HP1 score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `agent_rule_v7_dynamic` | 5 | 3.0 | 0.3455 | 0.4545 | 0.4000 | 0.7000 | 0.5244 |

与固定 `agent_rule_v7` 对比：

| 方法 | early recall | HP1 score |
|---|---:|---:|
| `agent_rule_v7` | 0.2000 | 0.5034 |
| `agent_rule_v7_dynamic` | 0.3455 | 0.5244 |

判断：

- dynamic early-slot 在严格 same-budget 下保持 `avg top-k=3.0`，没有扩大通信预算。
- early evidence recall 从 0.20 提升到 0.3455。
- HP1 multihop score 从 0.5034 提升到 0.5244。
- 这是目前 V7-agent-2 最明确的 agent 正信号。

## 4. Bandit 结果状态

`agent_bandit_v7_early` 已完成 9 个 run，覆盖 `early_coverage_weight=0.1/0.3/0.5` 与 seeds 0/1/2。official-style eval 中它被汇总为 `agent_bandit_v7`：

| 方法 | n | joint F1 | support title recall |
|---|---:|---:|---:|
| `agent_bandit_v7` | 9000 | 0.4441 | 0.7436 |
| `agent_rule_v7` | 20000 | 0.4437 | 0.7440 |

目前 bandit 在 fallback official eval 上没有明显超过 rule 主线；它更适合作为后续“reward 是否能稳定改变选择行为”的机制验证对象，而不是当前主结果候选。

## 5. Official-style Eval 结果与限制

当前 official eval 共完成 29/29：

- `v7agent2_ablation`: 20
- `v7agent2_bandit`: 9

汇总结果：

| 方法 | n | answer F1 | sp F1 | joint F1 | support title recall |
|---|---:|---:|---:|---:|---:|
| `agent_rule_v7` | 20000 | 0.8608 | 0.5060 | 0.4437 | 0.7440 |
| `agent_bandit_v7` | 9000 | 0.8599 | 0.5066 | 0.4441 | 0.7436 |

关键限制：

- 日志显示 `T5Tokenizer requires the SentencePiece library`。
- `FiDReader` 因 `sentencepiece` 缺失加载 `t5-base` 失败，自动退回 extractive fallback。
- 因此当前 official eval 只能称为 official-style fallback eval，不能称为严格 FiD/T5 reader 结果。
- 这也解释了 official 指标中方法差异很小：reader 没有真正利用 FiD 生成能力，更多反映 retrieval/support 选择的间接差异。

## 6. 当前研究判断

不要直接宣称 V7-agent-2 已在完整 official HotpotQA 上成功超过 baseline。更稳妥的表述是：

V7-agent-2 在 same-budget 条件下验证了 agentic selection 机制能够改变参数块选择行为，并在 HP1 strict diagnostic 上产生稳定正信号。其中，early prior 与 coverage replacement 是基础有效组件；dynamic early-slot 是目前最强的增益来源。当前 official-style fallback eval 尚未显示明显端到端优势，且严格 FiD/T5 reader 结果仍待补跑。

可以作为论文中的中间结论：

- Agent 方法不是通过增加通信预算获益，而是在同样 top-k=3 的 payload 约束下改变上传块结构。
- 对 early evidence 的显式建模能把原本为 0 或偏低的 early recall 拉起来。
- Dynamic slot allocation 进一步提升了 early evidence 覆盖，并带来 strict multihop score 改善。
- Memory 模块在当前设置中没有独立贡献，需要进一步强化与 reward delay/noise filtering 的耦合。

## 7. 后续优先级

1. 修复 FiD/T5 reader 环境：安装 `sentencepiece`，重跑 official eval。
2. 为 `v7agent2_dynamic` 补跑 official eval，使 dynamic 的 strict 正信号能进入端到端验证。
3. 增加 baseline official eval：至少加入 `hypernet_v6` 与 `adaptive_v6`，否则 official 表中 baseline 为 n=0，无法支持主论文对比。
4. 强化 memory：让 memory 不只是 EMA，而是参与 delayed reward smoothing、instability-aware exploration 与 block-level credit assignment。
5. 对 dynamic early-slot 做消融：固定 slot=1、dynamic slot、dynamic without rarity、dynamic without hard-query alignment。

## 8. 当前文件与报告位置

服务器最新报告：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2/实验分析报告/V7-agent-2/v7_agent2_complete_report_20260616.md`

服务器 dynamic strict 汇总：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2/实验分析报告/V7-agent-2/dynamic_strict_summary_agg.csv`

本地进展报告：

`/Users/iilab/ForAgent/实验分析报告/V7-agent-2/v7_agent2_experiment_progress_report_20260616.md`

