# V7-agent Hotpot Official Eval Summary

生成时间：2026-06-10T15:56:51

- 评估范围：`hp1_budget_aligned` 12 runs。
- 样本数：200 compact Hotpot examples per run。
- 注意：当前 evaluator 使用 retriever sentence ranking + heuristic answer，不是生成式 reader；因此主要看 support retrieval / joint proxy 是否传导。

| method | n_runs | answer EM | answer F1 | sp EM | sp F1 | joint EM | joint F1 | support title recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hypernet_v6 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7350 |
| adaptive_v6 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7350 |
| agent_bandit_v7 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7350 |
| agent_rule_v7 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7375 |

## 初步判断

- `agent_bandit_v7` joint F1 delta vs best baseline = +0.0000.
- `agent_rule_v7` joint F1 delta vs best baseline = +0.0000.

当前 official-style eval 没有显示 `agent_rule_v7` 的 strict selection gain 传导为明显 Hotpot joint F1 增益；各方法 answer/joint 指标几乎持平。更细的差别主要在 support_title_recall_at_k：`agent_rule_v7` 为 0.7375，baseline 为 0.7350。
