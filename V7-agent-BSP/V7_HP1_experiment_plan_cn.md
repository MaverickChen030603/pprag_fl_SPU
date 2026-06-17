# V7-HP1 HotpotQA 实验计划

## 核心目的

V7-HP1 立刻放弃旧的、对方法差异不敏感的数据设置，切换到 HotpotQA fullwiki 派生数据。HotpotQA 是多跳问答任务，包含问题、答案、supporting facts/context，更适合检验 agent memory、hard-query weighting、rarity/client embedding 是否真的改变选择行为，并进一步拉开与 baseline 的差距。

## 数据配置

- 上游训练数据：`FedE/select_data_hotpot_train_5000.json`
- 来源配置：`hotpot_qa/fullwiki/train`，5000 条样本
- 保留字段：`question`、`company`/answer、`page`、`reference`、`supporting_titles`
- RAG 评估预留：`hotpot_qa` validation，默认最多 1000 条

## Suite

- `smoke`: 1 run 起跑检查
- `hp1_multihop_hard`: Hotpot 多跳 hard-query 主实验
- `hp1_rare_bridge_tail`: Dirichlet alpha 0.1/0.05 的 rare bridge/tail 场景
- `hp1_budget_aligned`: top-k 预算对齐实验，防止把更大预算误判为 agent 能力
- `hp1_ablation_signal`: full/no_memory/no_hard_query/no_rarity，确认信号是否实质改变选择行为

## 自动化脚本

- `run_v7_hp1_all.sh smoke|full|hard|rare|budget|ablation`
- `check_v7_hp1_status.sh`
- `sync_github_v7_hp1.sh`

## 分析口径

HP1 strict 指标关注选择行为诊断：bridge block recall、early evidence recall、selection diversity、rare-client budget 和综合 `hp1_multihop_score`。它用于判断 agent 是否在 Hotpot 多跳压力下出现正信号；若 strict 指标正向，再接官方 answer/supporting-fact F1/EM 作为最终下游证明。
