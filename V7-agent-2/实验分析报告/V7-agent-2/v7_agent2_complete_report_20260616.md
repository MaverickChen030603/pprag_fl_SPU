# V7-agent-2 完整实验报告

生成时间：2026-06-16T14:10:16

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

## 2. Dynamic Strict 结果

```csv
method,agent_profile,n,avg_budget_topk,early_recall_mean,bridge_recall_mean,target_recall_mean,diversity_mean,hp1_score_mean
agent_rule_v7_dynamic,v7agent2_rule_dynamic,5,3.0,0.3454545454545454,0.454545454545454,0.3999999999999997,0.7,0.5243636363636361
```

## 3. Official Eval 结果（n=1000，FiD Reader）

```csv
method,n,answer_em_mean,answer_em_ci95_lo,answer_em_ci95_hi,answer_f1_mean,answer_f1_ci95_lo,answer_f1_ci95_hi,sp_em_mean,sp_em_ci95_lo,sp_em_ci95_hi,sp_f1_mean,sp_f1_ci95_lo,sp_f1_ci95_hi,joint_em_mean,joint_em_ci95_lo,joint_em_ci95_hi,joint_f1_mean,joint_f1_ci95_lo,joint_f1_ci95_hi,support_title_recall_at_k_mean,support_title_recall_at_k_ci95_lo,support_title_recall_at_k_ci95_hi
hypernet_v6,0,,,,,,,,,,,,,,,,,,,,,
adaptive_v6,0,,,,,,,,,,,,,,,,,,,,,
agent_bandit_v7,9000,0.859,0.8517,0.8659,0.8599,0.853,0.867,0.2087,0.2009,0.2168,0.5066,0.5007,0.5131,0.1817,0.1739,0.1904,0.4441,0.4367,0.4513,0.7436,0.7389,0.7487
agent_rule_v7,20000,0.8595,0.8545,0.8645,0.8608,0.8558,0.8655,0.208,0.2023,0.214,0.506,0.5016,0.5108,0.181,0.176,0.1863,0.4437,0.4389,0.4483,0.744,0.7407,0.7474
```

## 4. 结论

_待完整实验完成后填写；若 official eval 未显著提升，保留 V7-agent 的 strict diagnostic 结论边界。_

---

_本报告由 generate_v7agent2_report.py 自动生成，时间：2026-06-16T14:10:16_