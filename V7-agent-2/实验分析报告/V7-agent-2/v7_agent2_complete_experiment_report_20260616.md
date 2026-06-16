# V7-agent-2 完整实验报告

生成日期：2026-06-16  
项目名称：V7-agent-2  
项目路径：`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2`  
本地报告路径：`/Users/iilab/ForAgent/实验分析报告/V7-agent-2/v7_agent2_complete_experiment_report_20260616.md`

## 1. 实验背景与目的

V7 系列实验的核心目标是验证 Agentic Federated RAG 的有效性：在联邦训练中，不再把客户端视为只被动上传参数的节点，而是将 Client 升级为具备局部状态感知、历史记忆、策略规划和解释能力的 Client Agent。

V6/V7 早期结果暴露出一个关键问题：在严格 same-budget 约束下，也就是所有方法都只能上传同等 payload 或相同 top-k 参数块时，agent 方法很难在主结果上稳定拉开 baseline。V7-agent-2 的目的就是围绕这个痛点继续推进：

1. 在不扩大通信预算的前提下，让 agent 改变参数块选择行为。
2. 让 hard-query、early-evidence、rare-domain 相关参数块在 top-k 竞争中被优先保留。
3. 用 ablation 验证 early prior、coverage replacement、memory 等机制是否真正改变选择行为。
4. 用 dynamic early-slot 检验自适应 agent 策略能否比固定 rule 更强。
5. 用 official-style HotpotQA eval 观察 strict diagnostic 正信号能否传导到端到端问答指标。

## 2. 核心研究问题

本轮实验围绕以下问题展开：

1. Same-budget 下，agent 是否只是换了名字，还是实际改变了 upload mask？
2. Early evidence recall 能否从 V7-agent 的 0.20 进一步拉高？
3. Early prior 和 coverage replacement 是否是必要组件？
4. Memory 是否在当前任务设置下贡献了独立增益？
5. Bandit early reward 是否比 rule-based policy 更有效？
6. Dynamic early-slot 是否能在相同 top-k=3 预算下获得更高 multihop diagnostic score？
7. Strict diagnostic 的正信号是否能在 official-style HotpotQA eval 中体现？

## 3. 实验配置

### 3.1 数据集与任务

本轮实验使用 HotpotQA 风格的多跳问答任务，输入数据来自：

`/home/iiserver31/projects/FedE4RAG-main/FedE/select_data_hotpot_train_5000.json`

联邦任务目录：

`/home/iiserver31/projects/FedE4RAG-main/FedE/num5_dir_a03_imb00_ts0_v7hp1`

主要配置：

| 配置项 | 值 |
|---|---:|
| 客户端数量 | 5 |
| 数据划分 | Dirichlet |
| alpha | 0.3 |
| task seed | 0 |
| 本地 epoch | 1 |
| batch size | 8 |
| 联邦轮数 | 12 |
| 上传预算 | top-k=3 |
| 预算协议 | fixed same-budget |
| 主要模型 | BERT retriever / fedrag selective upload |

### 3.2 Same-budget 约束

所有 V7-agent-2 主实验都保持 `avg_budget_topk=3.0`，即每轮每客户端只上传 3 个参数块。这个约束是本实验的核心，因为它排除了“agent 只是多传了参数所以更好”的解释。

dynamic 实验的 strict diagnostic 结果显示：

| 方法 | avg top-k | budget std |
|---|---:|---:|
| `agent_rule_v7_dynamic` | 3.0 | 0.0 |

说明 dynamic early-slot 没有隐性扩大通信预算。

## 4. 方法设计

### 4.1 AgentMemory 与噪声过滤思想

V7-agent-2 延续 V7 的 agent memory 设计，目标是缓解下游 RAG reward 的高延迟和强噪声。核心思想是维护参数块级别的历史效用 EMA：

`utility_ema_b^t = rho * utility_ema_b^{t-1} + (1 - rho) * observed_reward_b^t`

同时引入 instability penalty，对跨轮次选择 mask 的剧烈波动进行惩罚，使历史记忆起到低通滤波作用。

不过本轮 ablation 显示，`no_memory` 与 full `agent_rule_v7` 的 strict diagnostic 指标基本一致。这说明当前 memory 机制虽然已接入，但在该 HP1 设置下尚未形成独立贡献。后续需要把 memory 更深地接入 delayed reward smoothing、credit assignment 和 exploration 稳定性。

### 4.2 Early Prior

HotpotQA 多跳任务对早期 evidence routing 较敏感。V7-agent-2 显式引入 early evidence prior，使 embeddings 与前几层 encoder blocks 在 hard-query 场景下获得额外评分补偿。

消融结果显示，去掉 early prior 后：

- early evidence recall 从 0.20 降到 0。
- HP1 score 从 0.5034 降到 0.4234。

这说明 early prior 不是装饰项，而是当前 agent 正信号的核心来源之一。

### 4.3 Coverage Replacement

Coverage replacement 的目的不是提高预算，而是在 top-k 已满时，用 early-evidence 相关块替换低优先级非 early 块，从而保证 strict same-budget 下仍能覆盖关键早期证据路径。

消融结果显示，去掉 coverage replacement 后：

- early evidence recall 从 0.20 降到 0。
- HP1 score 从 0.5034 降到 0.4476。

这说明 early prior 需要 coverage replacement 配合，才能真正影响最终 upload mask。

### 4.4 Bandit Early Reward

V7-agent-2 引入 `agent_bandit_v7_early`，扫描：

`early_coverage_weight = 0.1 / 0.3 / 0.5`

设计目的：

1. 让 bandit 不只关注局部 delta norm 或历史 utility。
2. 让 early evidence coverage 进入 reward/action 反馈。
3. 检查 bandit 是否能比 rule policy 更自适应。

目前 bandit 已完成 9 个 run，但在 official-style fallback eval 上与 rule 主线差异很小。因此 bandit 暂不作为主结果候选，更适合作为后续 reward 设计与 exploration 策略优化对象。

### 4.5 Dynamic Early-slot

Dynamic early-slot 是 V7-agent-2 中最重要的新机制。固定 `agent_rule_v7` 默认只保证有限 early coverage，而 dynamic policy 根据 query/client hardness 动态调整 early evidence slot 数量。

直觉是：

- 简单 query 不需要过多 early evidence slot。
- hard-query / multihop query 需要更多 early evidence coverage。
- 在 top-k=3 的预算中，slot 分配本身就是 agent planning 行为。

结果显示 dynamic policy 在 same-budget 下显著提升 strict diagnostic：

| 方法 | early recall | HP1 score |
|---|---:|---:|
| `agent_rule_v7` | 0.2000 | 0.5034 |
| `agent_rule_v7_dynamic` | 0.3455 | 0.5244 |

这是目前最明确的正信号。

## 5. 实验运行流程

V7-agent-2 建立为独立平行目录，避免干扰原始 V7-agent：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2`

主要脚本：

| 脚本 | 作用 |
|---|---|
| `run_v7agent2_ablation.sh` | 跑 full / no_prior / no_coverage / no_memory |
| `run_v7agent2_bandit.sh` | 跑 bandit early reward 权重扫描 |
| `run_v7agent2_official_eval.sh` | 跑 official-style Hotpot eval |
| `run_v7agent2_all.sh` | 全流程入口 |
| `run_hp1_strict_eval.py` | 计算 HP1 strict diagnostic |
| `analyze_ablation.py` | 汇总 ablation 与 Wilcoxon |
| `analyze_official_eval.py` | 汇总 official eval 与 bootstrap CI |
| `generate_v7agent2_report.py` | 自动生成报告 |

执行过程中的关键事件：

1. 首次 full pipeline 在报告生成时因 `tabulate` 缺失中断，已修复为无 `tabulate` 依赖。
2. official eval 初始路径指向旧 `hp1_budget_aligned`，导致 runs=0，已修复为扫描 `v7agent2_all`。
3. dynamic 初次在 GPU1 OOM，后改用 GPU0 单独重启，最终 5/5 完成。
4. official eval 中 `t5-base` FiD reader 因缺少 `sentencepiece` 加载失败，自动 fallback 到 extractive reader。

## 6. 实验完成情况

| 实验模块 | 完成状态 | run 数 | 输出 |
|---|---:|---:|---|
| `v7agent2_ablation` | 已完成 | 20/20 | strict diagnostic |
| `v7agent2_bandit` | 已完成 | 9/9 | strict + official fallback |
| `v7agent2_dynamic` | 已完成 | 5/5 | strict diagnostic |
| official-style eval | 已完成 | 29/29 | fallback reader |
| 总报告生成 | 已完成 | - | `v7_agent2_complete_report_20260616.md` |

## 7. 结果分析

### 7.1 Ablation 结果

| 方法 | n | early recall | HP1 score | bridge recall | avg top-k |
|---|---:|---:|---:|---:|---:|
| `agent_rule_v7` | 5 | 0.2000 | 0.5034 | 0.5985 | 3.0 |
| `agent_rule_v7_no_prior` | 5 | 0.0000 | 0.4234 | 0.6036 | 3.0 |
| `agent_rule_v7_no_coverage` | 5 | 0.0000 | 0.4476 | 0.6349 | 3.0 |
| `agent_rule_v7_no_memory` | 5 | 0.2000 | 0.5034 | 0.5985 | 3.0 |

主要结论：

- Early prior 和 coverage replacement 是必要组件。
- 它们不只是改变内部 score，而是实际改变 top-k upload mask。
- Memory 在当前配置下没有独立增益。
- full vs no-prior、full vs no-coverage 的 Wilcoxon p 值为 0.0625，方向稳定但未达到 0.05 显著性。

### 7.2 Dynamic Strict 结果

| 方法 | n | avg top-k | early recall | bridge recall | target recall | diversity | HP1 score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `agent_rule_v7_dynamic` | 5 | 3.0 | 0.3455 | 0.4545 | 0.4000 | 0.7000 | 0.5244 |

对比固定 rule：

| 方法 | early recall | HP1 score |
|---|---:|---:|
| `agent_rule_v7` | 0.2000 | 0.5034 |
| `agent_rule_v7_dynamic` | 0.3455 | 0.5244 |

解释：

Dynamic early-slot 降低了对固定 early slot 的依赖，使 hard-query 条件下更多 early evidence block 能进入 top-k。虽然 bridge recall 低于固定 rule，但 early recall 和 overall HP1 score 更高，说明 dynamic policy 在当前 scoring formula 下更符合 HP1 strict diagnostic 的目标。

### 7.3 Bandit Official-style 结果

| 方法 | n | answer F1 | sp F1 | joint F1 | support title recall |
|---|---:|---:|---:|---:|---:|
| `agent_rule_v7` | 20000 | 0.8608 | 0.5060 | 0.4437 | 0.7440 |
| `agent_bandit_v7` | 9000 | 0.8599 | 0.5066 | 0.4441 | 0.7436 |

解释：

- 在 fallback reader 下，bandit 与 rule 基本持平。
- joint F1 差异约 0.0004，没有实际意义。
- support title recall 也几乎一致。
- 当前结果不能证明 bandit 已优于 rule。

### 7.4 Official-style Eval 限制

虽然 official eval 已完成 29/29，但日志明确显示：

`T5Tokenizer requires the SentencePiece library`

因此 `FiDReader` 加载 `t5-base` 失败，实际使用 extractive fallback。当前 official eval 的定位应是：

official-style evaluation with fallback reader

而不是：

official FiD/T5 reader evaluation

这意味着：

- official eval 当前不能作为最终论文主结果。
- 它可以用于检查 retrieval/support 指标的大致趋势。
- 若要证明端到端 QA 提升，必须安装 `sentencepiece` 后重跑 FiD/T5 reader。

## 8. 当前结论

V7-agent-2 已经完成了从“机制设想”到“same-budget strict diagnostic 正信号”的闭环。最重要的发现是：

1. Agent selection 的改进不是来自通信预算扩张，所有关键实验均保持 top-k=3。
2. Early prior 与 coverage replacement 是拉起 early evidence recall 的必要机制。
3. Dynamic early-slot 是目前最有效的策略增强，使 early recall 从 0.20 提升到 0.3455，HP1 score 从 0.5034 提升到 0.5244。
4. Memory 模块当前没有独立贡献，需要重新设计 reward smoothing 与 block-level credit assignment。
5. Bandit early reward 当前没有超过 rule 主线。
6. Official fallback eval 尚未显示端到端显著优势，且由于缺少 `sentencepiece`，不能视作严格 FiD/T5 结果。

因此，当前最稳妥的论文表述是：

V7-agent-2 demonstrates that agentic upload planning can change same-budget parameter selection behavior and improve multihop-oriented strict diagnostics, especially through early-evidence-aware coverage and dynamic slot allocation. However, end-to-end HotpotQA gains under a true FiD/T5 reader remain to be validated.

## 9. 后续工作

### 9.1 立即优先级

1. 安装 `sentencepiece`，确认 `t5-base` reader 能正常加载。
2. 对 34 个 run 补跑真正 FiD/T5 official eval，特别是 `agent_rule_v7_dynamic`。
3. 补跑 baseline official eval：`hypernet_v6`、`adaptive_v6`。
4. 将 official eval 汇总表改成包含 baseline、rule、bandit、dynamic 四类方法。

### 9.2 方法优化方向

1. 强化 memory：让 EMA 不只是被动记录，而是参与 reward denoising、instability-aware exploration 和 delayed credit assignment。
2. 强化 bandit：让 UCB/Thompson sampling 的 reward 直接绑定 early evidence coverage 与 hard-query failure recovery。
3. 做 dynamic ablation：比较 fixed slot=1、dynamic slot、dynamic without hard-query、dynamic without rarity。
4. 做 rare-domain/hard-client 更强异构设置，让 agent rarity signal 有更大发挥空间。
5. 增加 per-query 行为分析，证明 agent 在 hard examples 上选择了不同块。

## 10. 文件索引

服务器完整报告：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2/实验分析报告/V7-agent-2/v7_agent2_complete_report_20260616.md`

服务器 dynamic 汇总：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2/实验分析报告/V7-agent-2/dynamic_strict_summary_agg.csv`

服务器进展报告：

`/home/iiserver31/projects/FedE4RAG-main/V7-agent-2/实验分析报告/V7-agent-2/v7_agent2_experiment_progress_report_20260616.md`

本地完整报告：

`/Users/iilab/ForAgent/实验分析报告/V7-agent-2/v7_agent2_complete_experiment_report_20260616.md`

GitHub 最新同步：

`c38d88d Add V7-agent-2 experiment progress report`

