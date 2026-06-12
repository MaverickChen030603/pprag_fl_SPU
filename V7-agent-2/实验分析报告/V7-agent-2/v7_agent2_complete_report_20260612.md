# V7-agent-2 完整实验报告

生成时间：2026-06-12T16:31:49

## 0. 执行摘要

V7-agent-2 在 V7-agent 基础上加入 FiD-style reader、三件套 ablation、official eval 统计检验、bandit early reward 和动态 early slot。

## 1. Ablation 结果

```csv
method,n,early_recall_mean,early_recall_std,hp1_score_mean,hp1_score_std,bridge_recall_mean,avg_budget_topk
hypernet_v6,0,,,,,,
agent_rule_v7,5,0.1999999999999998,0.0,0.5034327272727274,0.0091031946077251,0.5985454545454552,3.0
agent_rule_v7_no_prior,5,0.0,0.0,0.423418181818182,0.0115435846732043,0.6036363636363642,3.0
agent_rule_v7_no_coverage,5,0.0,0.0,0.4476145454545457,0.002304272711133,0.6349090909090915,3.0
agent_rule_v7_no_memory,5,0.1999999999999998,0.0,0.5034327272727274,0.0091031946077251,0.5985454545454552,3.0
```

## 2. Official Eval 结果（n=1000，FiD Reader）

```csv
method,n
hypernet_v6,0
adaptive_v6,0
agent_bandit_v7,0
agent_rule_v7,0
```

## 3. 结论

_待完整实验完成后填写；若 official eval 未显著提升，保留 V7-agent 的 strict diagnostic 结论边界。_

---

_本报告由 generate_v7agent2_report.py 自动生成，时间：2026-06-12T16:31:49_