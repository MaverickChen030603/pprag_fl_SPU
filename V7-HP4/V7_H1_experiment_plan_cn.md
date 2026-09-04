# V7-HP3 实验计划：Hard/Tail Agent Gap Search

## 目的

V7-HP3 是 V7 的平行分支，不覆盖 V7 结果。它的目标不是直接宣称 agent 成功，而是在 V7 下游指标饱和的问题之后，构造一个更能暴露 agent 行为差异的非饱和诊断环境，检验 agent 是否能在 hard-query、rare-domain / tail-client、memory / rarity signal 场景中拉开与 `hypernet_v6`、`adaptive_v6` 的差距。

## 核心假设

1. 标准 RAG 指标在 V7 中容易全满分，导致真实选择差异被吞掉。
2. 如果把评估焦点转到 hard block、tail block、client/query-aware selection diversity，agent 的记忆、稀有性和 hard-query 信号应当改变选择行为。
3. `agent_oracle_v7hp2` 只作为 upper bound，用于判断动作空间是否具备可拉开差距的上限，不作为公平主方法。

## 方法组

- `hypernet_v6`: V6/V7 强基线。
- `adaptive_v6`: 自适应预算基线。
- `agent_tail_v7hp2`: hard-client 与 tail-block aware，使用 client/query/rarity 信号调整 block 选择与 top-k。
- `agent_memory_v7hp2`: 使用 history / diversity memory 改变选择轨迹，避免每轮重复相同 block。
- `agent_oracle_v7hp2`: tail/hard upper-bound 诊断策略。

## Suite

- `h1_hardquery_non_saturated`: hard-query scale 加强，目标是让 baseline 不再在诊断指标上满分。
- `h1_rare_domain_tail`: 更低 Dirichlet alpha，强化 rare-domain / hard-client 异质性。
- `h1_action_space`: 比较 H1 agent 动作空间本身是否改变预算和 block selection。
- `h1_ablation`: 对 `agent_tail_v7hp2` 关闭 memory、hard-query、rarity 相关信号，确认每个信号是否真的改变选择行为。

默认 full H1 共 69 个 upstream run：15 + 30 + 12 + 12。

## 指标解释

V7-HP3 新增 `run_h1_strict_eval.py`，从 upstream `round_logs.csv` 直接计算非饱和诊断指标：

- `avg_budget_topk_h1`: 平均上传 block 数，检查预算是否对齐。
- `tail_client_budget_h1`: hard/tail client 的预算响应。
- `hard_block_recall_h1`: hard block 被选择比例。
- `tail_block_recall_h1`: tail block 被选择比例。
- `selection_diversity_h1`: 选择多样性。
- `h1_non_saturated_score`: 综合诊断分数。

这些不是标准 RAG F1/EM，而是为判断 agent 是否改变选择行为而设计的代理指标。标准 RAG eval 可通过 `RUN_STANDARD_RAG=1 ./run_v7_hp3_all.sh full` 额外开启。

## 一键运行

```bash
./run_v7_hp3_all.sh smoke
./run_v7_hp3_all.sh full
```

查看状态：

```bash
./check_v7_hp3_status.sh
```

同步代码：

```bash
./sync_github_v7_hp1.sh
```
