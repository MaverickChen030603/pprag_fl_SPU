# V7 Experiment Set: Agentic Federated RAG

更新日期：2026-05-27  
建议项目目录：`/Users/iilab/PPRAG_FL/FedE4RAG-main/FedE4RAG-main/V7`  
建议服务器目录：`/home/iiserver31/projects/FedE4RAG-main/V7`

## 1. V7 总体定位

V7 是在 V3-V6 基础上的研究方向拓展，目标是把传统联邦学习中的被动 client 进一步建模为具有局部观察、历史记忆、策略选择和解释能力的 client agent。

V7 不应仅把 client 改名为 agent，而应明确提出一个新的研究问题：

> 在 Federated RAG 中，client agent 能否在严格相同通信预算下，更主动、更精准地选择对下游检索与问答最有价值的参数块，从而优于传统选择性上传策略？

V7 与前序版本的关系：

- V3：证明选择性上传在联邦 RAG 中是可行的，能显著降低通信量并保持下游效果。
- V4：加入 downstream-aware、hard-query、hard-client 和 adaptive budget，在困难场景下更稳，但有预算膨胀问题。
- V5：收紧预算扩张，并加入 budget-aligned 比较，但主结果仍未明显超过 V3。
- V6：聚焦严格 same-budget comparison，验证固定通信预算下是否能更精准地上传有价值参数块。
- V7：在 V6 的 same-budget 框架上，把 client 从被动选择器升级为 agentic decision maker，引入状态感知、记忆、策略规划、反馈学习和可解释决策。

## 2. 核心科研问题

V7 的核心问题是：

1. 在相同通信预算下，agentic client 是否比 random、delta_norm、hypernet_v6、adaptive_v6 更能提升下游 RAG 效果？
2. 在强异构、hard-query、rare-domain 场景下，agent 是否能更合理地把通信预算分配给真正有价值的 client 和参数块？
3. agent 的记忆、下游反馈、hard-query signal、client rarity signal 分别贡献多少？
4. agent 的决策是否具有可解释性，能否说明为什么某些 client/block 被优先上传？
5. agentic decision 的额外计算成本是否小于通信节省和下游收益带来的价值？

## 3. 核心假设

### H1: Agentic Upload Superiority

在严格相同通信预算下，具备状态感知和历史记忆的 client agent 能比静态规则或单轮选择器更好地选择参数块，从而获得更高的 downstream RAG utility。

### H2: Hard Query Sensitivity

在 hard-query setting 中，agent 能利用失败查询、检索困难度和历史奖励信号，将更多上传预算分配给 hard-query 相关参数块。

### H3: Heterogeneity Robustness

在 non-IID、rare-domain、hard-client 场景下，agent 能更好地识别高价值但低频的 client contribution，从而改善强异构条件下的整体检索效果和 client fairness。

### H4: Memory Matters

相比只看当前轮状态的选择器，带有历史记忆的 agent 能更稳定地估计 block utility，减少上传抖动，提高 payload-normalized performance。

### H5: Agentic Explanation Validity

agent 的结构化解释应与实际贡献信号一致。例如，被解释为 hard-query-related 的 block 应在 hard-query subset 上产生更明显的 downstream improvement。

## 4. Agentic Client 定义

V7 中的 client agent 是一个联邦参与节点，其行为不只包括本地训练和上传参数，还包括：

1. 观察本地训练与下游反馈状态。
2. 维护历史上传和收益记忆。
3. 估计参数块的下游效用。
4. 在固定通信预算内选择上传动作。
5. 根据 server/global hint 调整策略。
6. 输出结构化解释。

### 4.1 Agent State

每个 client agent 在第 `t` 轮维护状态：

```text
s_i^t = {
  local_loss,
  local_loss_delta,
  local_update_norm,
  block_delta_norms,
  block_gradient_statistics,
  local_domain_profile,
  local_query_profile,
  hard_query_alignment,
  retrieval_failure_signal,
  previous_upload_mask,
  previous_reward,
  historical_block_utility,
  client_rarity_score,
  current_budget,
  global_round_context
}
```

### 4.2 Agent Memory

每个 client agent 维护轻量历史记忆：

```text
M_i^t = {
  uploaded_blocks_history,
  payload_history,
  downstream_reward_history,
  hard_query_reward_history,
  block_selection_frequency,
  block_success_score,
  client_contribution_score,
  strategy_mode_history
}
```

记忆更新建议使用指数滑动平均：

```text
utility_ema_b^t = rho * utility_ema_b^{t-1} + (1 - rho) * observed_reward_b^t
```

其中 `rho` 建议取 `0.7` 或 `0.9`，后续在 ablation 中比较。

### 4.3 Agent Action

第一阶段建议动作空间保持克制：

```text
a_i^t = {
  upload_mask_i^t,
  strategy_mode_i^t,
  optional_budget_adjustment_i^t,
  explanation_i^t
}
```

主要动作：

- `select_blocks`: 在固定 top-k 或 payload budget 下选择上传哪些参数块。
- `choose_strategy_mode`: 在不同策略模式之间切换。
- `adjust_local_budget`: 在全局预算约束内微调本 client 的本轮预算。
- `explain_decision`: 给出选择原因。

建议 V7 第一篇论文的主实验只使用 `select_blocks` 和 `choose_strategy_mode`，把 `adjust_local_budget` 放到扩展实验，避免问题过宽。

### 4.4 Agent Reward

奖励应围绕 downstream utility，而不是只围绕本地 loss：

```text
r_i^t =
  w1 * DeltaRetrievalScore
+ w2 * DeltaQAQuality
+ w3 * HardQueryGain
+ w4 * RareDomainGain
- w5 * PayloadCost
- w6 * InstabilityPenalty
```

其中：

- `DeltaRetrievalScore`: Recall@k / MRR / nDCG 的提升。
- `DeltaQAQuality`: EM / F1 / answer accuracy 的提升。
- `HardQueryGain`: hard-query subset 上的提升。
- `RareDomainGain`: rare-domain query subset 上的提升。
- `PayloadCost`: 上传通信成本。
- `InstabilityPenalty`: 选择 mask 过度抖动或下游波动惩罚。

如果 downstream evaluation 无法每轮完整运行，可使用 proxy reward：

```text
proxy_reward =
  alpha * hard_query_alignment
+ beta  * retrieval_proxy_gain
+ gamma * client_rarity_score
+ eta   * historical_block_success
- lambda * normalized_payload
```

## 5. V7 方法族设计

V7 建议包含三类 agentic 方法，从易实现到强方法逐步推进。

### 5.1 agent_rule_v7

规则型 agent，是 V7 的最小可落地版本。

每个参数块 `b` 的上传分数：

```text
score_i,b =
  alpha * norm_delta_i,b
+ beta  * downstream_proxy_i,b
+ gamma * hard_query_alignment_i,b
+ eta   * client_rarity_i
+ mu    * historical_success_i,b
- lambda * redundancy_i,b
- tau    * instability_penalty_i,b
```

选择分数最高且满足预算的 top-k blocks。

建议策略模式：

- `stability_focused`: 更重视 historical_success 与低抖动。
- `hard_query_focused`: 更重视 hard_query_alignment。
- `diversity_focused`: 更重视 client_rarity 与低 redundancy。
- `communication_focused`: 更严格惩罚 payload。

默认权重建议：

```text
alpha  = 0.25
beta   = 0.25
gamma  = 0.20
eta    = 0.10
mu     = 0.15
lambda = 0.10
tau    = 0.05
```

权重可在 `v7_ablation_signal` 中逐个移除。

### 5.2 agent_bandit_v7

上下文 bandit agent，把每个参数块视为可选择 arm，利用历史 reward 更新 block utility。

推荐两种实现：

1. `ucb_agent`

```text
score_i,b = mean_reward_i,b + c * sqrt(log(t) / (n_i,b + 1))
```

2. `thompson_agent`

对每个 block 维护 reward posterior，按采样值选择 top-k。

该方法的优势是更像 agentic learning，同时实现成本低于强化学习。

### 5.3 agent_policy_v7

学习型 policy agent，输入 client/block features，输出 block selection probability。

输入特征：

```text
x_i,b^t = [
  norm_delta_i,b,
  downstream_proxy_i,b,
  hard_query_alignment_i,b,
  historical_success_i,b,
  selection_frequency_i,b,
  client_rarity_i,
  current_budget_i,
  round_index,
  suite_type_embedding
]
```

输出：

```text
p_i,b^t = PolicyNet(x_i,b^t)
```

选择方式：

- training: Gumbel-TopK 或 stochastic top-k。
- evaluation: deterministic top-k。

训练方式：

- imitation learning：模仿 oracle 或历史最佳 block。
- offline RL / contextual bandit：使用历史 run 的 downstream reward。
- online fine-tuning：在当前 suite 中少量更新。

建议第一阶段优先实现 imitation + bandit，不建议一开始做复杂 RL。

### 5.4 agent_llm_planner_v7

LLM-assisted agent 不直接选择具体参数块，而是做高层策略规划。

LLM 输入：

```text
{
  suite_type,
  current_round,
  global_budget_status,
  client_heterogeneity_summary,
  hard_query_failure_summary,
  previous_round_downstream_change,
  current_strategy_statistics
}
```

LLM 输出结构化策略：

```json
{
  "strategy_mode": "hard_query_focused",
  "alpha_delta": 0.20,
  "beta_downstream": 0.30,
  "gamma_hard_query": 0.30,
  "eta_rarity": 0.10,
  "mu_memory": 0.10,
  "explanation": "Hard-query failures concentrate on rare-domain clients, so prioritize blocks aligned with those queries."
}
```

低层 block selection 仍由 `agent_rule_v7` 或 `agent_policy_v7` 执行。

这样设计可以避免 LLM 直接控制底层参数上传带来的不稳定和成本问题。

## 6. V7 Baselines

V7 必须与 V6 主方法同台比较。

必选 baseline：

- `full_upload`
- `random`
- `delta_norm`
- `hypernet_v5`
- `hypernet_v6`
- `adaptive_v6`

可选 baseline：

- `oracle_topk_proxy`: 使用完整 downstream proxy 的近似 oracle。
- `no_agent_rule`: 只用单轮 score，无 memory、无 mode。
- `server_only_selector`: server 根据聚合统计选择 block，client 不具备 agentic decision。

关键比较不应是 V7 比 weak baseline 好，而应是：

```text
agent_rule_v7 / agent_bandit_v7 / agent_policy_v7
vs
hypernet_v6 / adaptive_v6
under exactly the same payload budget
```

## 7. V7 Experiment Suites

建议 V7 包含 8 个 suite。

### 7.1 v7_main

目的：

验证标准设置下，agentic client 是否能在相同通信预算下保持或提升 RAG 效果。

方法：

- `random`
- `delta_norm`
- `hypernet_v6`
- `adaptive_v6`
- `agent_rule_v7`
- `agent_bandit_v7`
- `agent_policy_v7`

种子：

- `seed = 0, 1, 2`

top-k：

- `topk = 3`
- 可追加 `topk = 5`

输出：

- upstream training logs
- payload logs
- downstream RAGTest logs
- suite report
- full pipeline report

### 7.2 v7_budget_aligned

目的：

V7 最重要的 suite，严格回答同预算下 agent 是否更优。

约束：

- 所有方法使用完全相同的 total payload budget。
- 每轮上传预算相同或按同一 schedule 分配。
- 禁止 agent_llm_planner 通过扩预算获得优势。

方法：

- `random`
- `delta_norm`
- `hypernet_v6`
- `adaptive_v6`
- `agent_rule_v7`
- `agent_bandit_v7`
- `agent_policy_v7`

核心指标：

- downstream score under same payload
- hard-query score under same payload
- payload-normalized utility
- improvement per MB

预期结果：

agent 方法不一定显著降低 payload，但应在同 payload 下提高 downstream utility。

### 7.3 v7_heterogeneity

目的：

验证 agent 在 non-IID、hard-client、rare-domain 场景下是否更能识别高价值 client。

设置：

- 增大 client data heterogeneity。
- 构造 rare-domain clients。
- 构造 high-loss but high-value clients。

方法：

- `random`
- `delta_norm`
- `adaptive_v6`
- `agent_rule_v7`
- `agent_bandit_v7`
- `agent_policy_v7`

额外指标：

- rare-domain Recall@k
- hard-client contribution rate
- client selection entropy
- client fairness score
- upload budget concentration ratio

### 7.4 v7_hardquery

目的：

验证 agent 是否能针对 hard-query 失败模式调整上传策略。

设置：

- 使用 hard-query subset。
- 记录 query-level retrieval failures。
- 将 hard-query feedback 转为 agent state。

方法：

- `hypernet_v6`
- `adaptive_v6`
- `agent_rule_v7`
- `agent_bandit_v7`
- `agent_policy_v7`

额外指标：

- hard-query Recall@k
- hard-query MRR
- hard-query answer F1
- hard-query gain per payload
- selected blocks 与 hard-query alignment 的相关性

### 7.5 v7_ablation_signal

目的：

分析 agent 各信号的贡献。

以 `agent_rule_v7` 和 `agent_policy_v7` 为主，比较：

- full agent
- no downstream proxy
- no hard-query signal
- no client rarity
- no memory
- no redundancy penalty
- no instability penalty
- current-round only

关键问题：

如果去掉 memory 或 hard-query signal，hard-query 和 heterogeneity 场景是否明显下降？

### 7.6 v7_ablation_agent_level

目的：

证明 V7 的提升不是来自简单调参，而是来自 agentic design。

比较：

- passive client selector
- reactive client agent: only current state
- memory agent: current state + history
- planning agent: memory + strategy mode
- llm-planner agent: planning agent + LLM high-level mode selection

预期：

从 passive 到 planning agent 应逐步提升；LLM planner 不一定主指标最强，但应在解释性和复杂 setting 下更有优势。

### 7.7 v7_cost_efficiency

目的：

评估 agent 引入后的额外成本是否合理。

记录：

- local compute overhead
- selector/policy inference time
- memory update time
- LLM call count
- LLM latency
- LLM token cost
- wall-clock time
- payload saved
- score gained per compute cost

核心结论应回答：

agentic decision 是否用少量计算换来了更高通信效率和下游收益？

### 7.8 v7_explain

目的：

解释 agent 为什么选择某些 client/block，并验证解释是否与真实贡献一致。

输出：

- per-round selected clients
- per-round selected blocks
- strategy mode
- top contributing signals
- explanation text
- hard-query alignment evidence
- historical utility evidence

建议生成报告：

- `v7_agent_decision_trace_cn.md`
- `v7_explain_case_study_cn.md`

案例分析：

1. agent 为什么选择 rare-domain client。
2. agent 为什么放弃 high-delta 但 low-downstream-utility block。
3. agent 如何从 stability-focused 切换到 hard-query-focused。
4. agent 选择与 downstream gain 的一致性。

## 8. 推荐完整运行矩阵

### 8.1 主实验矩阵

```text
suite:
  - v7_main
  - v7_budget_aligned
  - v7_heterogeneity
  - v7_hardquery

method:
  - random
  - delta_norm
  - hypernet_v6
  - adaptive_v6
  - agent_rule_v7
  - agent_bandit_v7
  - agent_policy_v7

seed:
  - 0
  - 1
  - 2

topk:
  - 3
```

主实验 run 数：

```text
4 suites * 7 methods * 3 seeds * 1 topk = 84 upstream runs
84 downstream RAG evaluation runs
```

### 8.2 消融实验矩阵

```text
suite:
  - v7_ablation_signal
  - v7_ablation_agent_level

method variants:
  - full_agent
  - no_downstream_proxy
  - no_hard_query
  - no_client_rarity
  - no_memory
  - no_redundancy_penalty
  - no_instability_penalty
  - current_round_only

seed:
  - 0
  - 1
  - 2
```

消融 run 数：

```text
2 suites * 8 variants * 3 seeds = 48 upstream runs
48 downstream RAG evaluation runs
```

### 8.3 成本与解释实验矩阵

```text
suite:
  - v7_cost_efficiency
  - v7_explain

method:
  - agent_rule_v7
  - agent_bandit_v7
  - agent_policy_v7
  - agent_llm_planner_v7

seed:
  - 0
  - 1
```

成本解释 run 数：

```text
2 suites * 4 methods * 2 seeds = 16 upstream runs
16 downstream RAG evaluation runs
```

### 8.4 总体规模

建议 V7 完整实验规模：

```text
84 + 48 + 16 = 148 upstream runs
148 downstream RAG evaluation runs
```

如果算力紧张，建议 first-pass 缩减为：

```text
v7_main + v7_budget_aligned + v7_hardquery
methods: random, delta_norm, hypernet_v6, adaptive_v6, agent_rule_v7, agent_bandit_v7
seeds: 0, 1, 2

3 suites * 6 methods * 3 seeds = 54 upstream runs
54 downstream runs
```

## 9. 评价指标

### 9.1 通信指标

- `avg_payload`
- `total_payload`
- `payload_ratio`
- `payload_per_round`
- `payload_variance`
- `uploaded_blocks_count`
- `client_upload_frequency`
- `budget_violation_count`

### 9.2 下游 RAG 指标

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `MRR`
- `nDCG`
- `answer_EM`
- `answer_F1`
- `RAGTest_score`

### 9.3 通信效率指标

建议新增：

```text
UtilityPerPayload = DownstreamScore / TotalPayload
DeltaUtilityPerPayload = (DownstreamScore_method - DownstreamScore_baseline) / TotalPayload
HardQueryGainPerPayload = HardQueryGain / TotalPayload
```

### 9.4 Agent 指标

- `strategy_mode_distribution`
- `memory_hit_rate`
- `selected_block_historical_success`
- `hard_query_alignment_mean`
- `client_rarity_selected_mean`
- `selection_stability`
- `selection_entropy`
- `explanation_signal_consistency`

### 9.5 公平性与异构性指标

- `rare_domain_gain`
- `hard_client_gain`
- `client_fairness_score`
- `budget_concentration_ratio`
- `low_frequency_client_selection_rate`

## 10. Same-Budget Protocol

V7 必须继承并强化 V6 的 same-budget protocol。

### 10.1 预算约束

每个方法必须满足：

```text
TotalPayload(method) <= TotalPayloadBudget
```

并记录：

```text
BudgetGap = TotalPayloadBudget - TotalPayload(method)
```

若不同方法 payload 有轻微差异，需要使用插值或归一化指标报告：

```text
ScoreAtFixedPayload
UtilityPerPayload
```

### 10.2 禁止隐性预算扩张

V7 agent 不允许通过以下方式获得不公平优势：

- 更多上传轮数。
- 更大 top-k。
- 额外上传未计入 payload 的 metadata。
- LLM planner 使用额外下游标签但 baseline 不可访问。
- 使用未来轮次的 downstream result。

### 10.3 Agent 可用信息边界

允许 agent 使用：

- 当前轮本地训练状态。
- 历史轮次自身上传记录。
- 历史 reward 或 proxy reward。
- server 广播的全局统计。
- 已公开的 hard-query summary。

不允许 agent 使用：

- 当前轮未来 downstream evaluation。
- 其他 client 的私有原始数据。
- 未对 baseline 公平开放的测试标签。

## 11. 预期结果与判定标准

### 11.1 强成功标准

满足以下条件可以认为 V7 强成功：

1. `agent_policy_v7` 或 `agent_bandit_v7` 在 `v7_budget_aligned` 中显著超过 `hypernet_v6` 和 `adaptive_v6`。
2. 在 `v7_hardquery` 中 hard-query Recall/MRR/F1 明显提升。
3. 在 `v7_heterogeneity` 中 rare-domain 或 hard-client gain 明显提升。
4. payload 没有超过 V6 same-budget 约束。
5. ablation 显示 memory、hard-query signal、downstream proxy 至少有两个信号是关键贡献项。

### 11.2 中等成功标准

如果主指标提升有限，但满足以下条件，也可以作为论文扩展：

1. agent 方法在 hard-query 或 heterogeneity 场景稳定优于 baseline。
2. payload-normalized utility 优于 hypernet_v6。
3. explain analysis 能清楚展示 agent 更合理地分配预算。
4. cost overhead 可控。

### 11.3 失败但有价值的发现

如果 V7 没有超过 V6，也仍可形成负结果分析：

1. 在当前 Federated RAG 设置下，agentic memory 不一定优于强 hypernet selector。
2. 下游 reward 延迟和噪声可能限制 agent learning。
3. client-level agent 需要更强 query-level feedback 才能发挥作用。
4. LLM planner 的高层策略不一定转化为底层参数选择收益。

## 12. 实现建议

### 12.1 目录结构

建议在项目中新增：

```text
V7/
  agent/
    state.py
    memory.py
    reward.py
    policy_rule.py
    policy_bandit.py
    policy_network.py
    llm_planner.py
    explain.py
  configs/
    v7_main.yaml
    v7_budget_aligned.yaml
    v7_heterogeneity.yaml
    v7_hardquery.yaml
    v7_ablation_signal.yaml
    v7_ablation_agent_level.yaml
    v7_cost_efficiency.yaml
    v7_explain.yaml
  run_experiment_suite.py
  run_all_rag_eval.py
  finalize_pipeline.py
  report_utils.py
  README_V7.md
```

### 12.2 与 V6 复用

优先复用 V6：

- training loop
- downstream RAGTest
- output directory convention
- suite report
- full pipeline report
- same-budget logic
- hard-query and heterogeneity setting

只新增 agent selector 模块，避免重写整个 pipeline。

### 12.3 日志字段

每轮建议记录：

```json
{
  "round": 1,
  "client_id": "client_0",
  "method": "agent_rule_v7",
  "strategy_mode": "hard_query_focused",
  "selected_blocks": [3, 7, 12],
  "block_scores": {
    "3": 0.81,
    "7": 0.76,
    "12": 0.72
  },
  "score_components": {
    "delta": 0.21,
    "downstream_proxy": 0.24,
    "hard_query": 0.19,
    "rarity": 0.08,
    "memory": 0.12,
    "redundancy_penalty": -0.03
  },
  "payload": 0.19,
  "memory_summary": {
    "avg_historical_success": 0.63,
    "selection_stability": 0.71
  },
  "explanation": "Selected blocks with high hard-query alignment and stable historical downstream gain."
}
```

## 13. 报告产物

V7 完成后建议生成：

```text
V7/v7_complete_experiment_record_cn.md
V7/v7_complete_experiment_analysis_cn.md
V7/v7_agent_design_cn.md
V7/v7_agent_decision_trace_cn.md
V7/v7_budget_aligned_report_cn.md
V7/v7_hardquery_case_study_cn.md
V7/v7_heterogeneity_case_study_cn.md
```

总报告命名建议：

```text
full_pipeline_all_v7_YYYY-MM-DD_HH-MM-SS
```

## 14. 论文叙事建议

V7 的论文叙事应避免泛泛地说 AI agent，而要强调 Federated RAG 的特殊性：

1. Federated RAG 中不同 client 的知识域和 query relevance 差异很大。
2. 通信预算有限，因此上传决策本质上是一个 budgeted decision-making problem。
3. 传统 client 是被动训练节点，无法根据历史下游收益主动调整策略。
4. Agentic client 能利用本地状态、历史记忆和下游反馈，在固定预算下主动选择更有价值的上传内容。
5. 实验表明，agentic decision 在 hard-query 和 heterogeneous settings 中尤其有效。

建议论文贡献写法：

```text
1. We introduce Agentic Federated RAG, a framework that upgrades passive federated clients into state-aware and memory-augmented client agents for communication-efficient RAG training.
2. We propose agentic upload policies that select parameter blocks under strict communication budgets using downstream utility, hard-query alignment, client rarity, and historical reward signals.
3. We establish a same-budget evaluation protocol to compare agentic clients with strong selective-upload baselines including hypernetwork and adaptive selectors.
4. We provide extensive experiments and explanations showing when and why agentic clients improve RAG utility, especially under heterogeneous and hard-query settings.
```

## 15. 推荐推进顺序

### Phase 1: V7-lite

目标：快速验证方向。

实现：

- `agent_rule_v7`
- memory
- downstream proxy
- hard-query signal
- `v7_main`
- `v7_budget_aligned`
- `v7_hardquery`

成功后再进入 Phase 2。

### Phase 2: V7-learning

目标：证明 agent 不只是规则加权。

实现：

- `agent_bandit_v7`
- `agent_policy_v7`
- `v7_heterogeneity`
- `v7_ablation_signal`
- `v7_ablation_agent_level`

### Phase 3: V7-LLM

目标：连接 AI agent 热点，但控制风险。

实现：

- `agent_llm_planner_v7`
- strategy mode generation
- explanation generation
- `v7_cost_efficiency`
- `v7_explain`

LLM planner 应作为增强模块，不应作为唯一主方法。

## 16. 最小可投稿版本

如果时间有限，最小可投稿版本建议包含：

方法：

- `random`
- `delta_norm`
- `hypernet_v6`
- `adaptive_v6`
- `agent_rule_v7`
- `agent_bandit_v7`

suite：

- `v7_main`
- `v7_budget_aligned`
- `v7_heterogeneity`
- `v7_hardquery`
- `v7_ablation_signal`
- `v7_explain`

种子：

- `0, 1, 2`

核心结论必须围绕：

```text
Same communication budget, better downstream RAG utility.
```

## 17. 当前最重要提醒

V7 的最大价值不在于“用了 agent 这个热门词”，而在于把 Federated RAG 的通信选择问题重新定义为：

> 多个具备局部知识、历史记忆和下游反馈的 client agents，在隐私和通信预算约束下进行协作式上传决策。

因此，V7 的实验设计必须始终围绕三个关键词：

```text
same-budget
downstream utility
agentic decision
```

只要这三点站住，V7 就能自然继承 V3-V6，并把你的科研方向从 communication-efficient federated RAG 推进到 agentic federated RAG。
