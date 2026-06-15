# V7-agent-2 完整实验报告

生成时间：2026-06-15T18:32:37

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
method,n,answer_em_mean,answer_em_ci95_lo,answer_em_ci95_hi,answer_f1_mean,answer_f1_ci95_lo,answer_f1_ci95_hi,sp_em_mean,sp_em_ci95_lo,sp_em_ci95_hi,sp_f1_mean,sp_f1_ci95_lo,sp_f1_ci95_hi,joint_em_mean,joint_em_ci95_lo,joint_em_ci95_hi,joint_f1_mean,joint_f1_ci95_lo,joint_f1_ci95_hi,support_title_recall_at_k_mean,support_title_recall_at_k_ci95_lo,support_title_recall_at_k_ci95_hi
hypernet_v6,0,,,,,,,,,,,,,,,,,,,,,
adaptive_v6,0,,,,,,,,,,,,,,,,,,,,,
agent_bandit_v7,9000,0.859,0.8526,0.8661,0.8599,0.8525,0.8672,0.2087,0.2,0.2168,0.5066,0.5001,0.5123,0.1817,0.1742,0.1893,0.4441,0.4374,0.4515,0.7436,0.7383,0.7492
agent_rule_v7,20000,0.8595,0.8547,0.8642,0.8608,0.8555,0.8653,0.208,0.2024,0.2137,0.506,0.5017,0.5102,0.181,0.1758,0.1864,0.4437,0.4392,0.4483,0.744,0.7407,0.7476
```

## 3. 结论

_待完整实验完成后填写；若 official eval 未显著提升，保留 V7-agent 的 strict diagnostic 结论边界。_

---

_本报告由 generate_v7agent2_report.py 自动生成，时间：2026-06-15T18:32:37_