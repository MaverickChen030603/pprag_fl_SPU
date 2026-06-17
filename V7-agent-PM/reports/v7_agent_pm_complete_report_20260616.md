# V7-agent-PM 完整实验报告

生成时间：2026-06-17T08:32:04

## 1. 实验目的

本轮实验在 V7-agent-2 基础上验证 Planning-Memory Agent 是否能在 fixed same-budget top-k=3 下超过 `agent_rule_v7_dynamic`，并检查 memory/failure/rarity/instability 是否真实改变 upload block selection。

## 2. 与 V7-agent-2 的关系

V7-agent-2 已证明 dynamic early-slot 在 strict diagnostic 上有正信号。本轮 V7-agent-PM 新增 PM scoring、memory ablation、dynamic planning ablation、bandit slot policy、true FiD/T5 reader 检查与 subgroup/per-query 分析。

## 3. 方法设计

PM score 使用 base delta、early prior、coverage gain、utility EMA、failure recovery、rarity、instability penalty。所有策略必须保持 top-k=3。

## 4. Same-budget 约束确认

见 strict diagnostic 表中的 `avg_topk` 与 `budget_std`。

## 5. Strict Diagnostic 结果

```csv
method,agent_profile,n,early_recall,bridge_recall,target_recall,diversity,hp1_score,avg_topk,budget_std
adaptive_v6,hp1_baseline_adaptive_v6,5,0.0,0.588363636363637,0.2941818181818184,0.42,0.4134618181818183,3.0,0.0
agent_dynamic_no_bridge_guard,v7pm_dynamic_no_bridge_guard,5,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.6399999999999999,0.5094909090909089,3.0,0.0
agent_dynamic_no_hardness,v7pm_dynamic_no_hardness,5,0.1999999999999998,0.5956363636363642,0.3978181818181815,0.48,0.4902981818181819,3.0,0.0
agent_dynamic_no_rarity,v7pm_dynamic_no_rarity,5,0.3454545454545454,0.4509090909090903,0.3981818181818179,0.6199999999999999,0.5069454545454544,3.0,0.0
agent_dynamic_slot,v7pm_dynamic_slot,5,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.6399999999999999,0.5094909090909089,3.0,0.0
agent_fixed_slot_0,v7pm_fixed_slot_0,5,0.0,0.6312727272727279,0.3156363636363639,0.5,0.4461963636363639,3.0,0.0
agent_fixed_slot_1,v7pm_fixed_slot_1,5,0.1999999999999998,0.5956363636363642,0.3978181818181815,0.48,0.4902981818181819,3.0,0.0
agent_fixed_slot_2,v7pm_fixed_slot_2,5,0.3999999999999997,0.399272727272727,0.3996363636363633,0.5599999999999999,0.4917163636363634,3.0,0.0
agent_fixed_slot_3,v7pm_fixed_slot_3,5,0.6000000000000006,0.1999999999999998,0.3999999999999997,0.5599999999999999,0.4760000000000001,3.0,0.0
agent_pm_bandit_slot,v7pm_bandit_slot_ucb_proxy,5,0.3636363636363635,0.4290909090909086,0.3963636363636361,0.8400000000000001,0.548072727272727,3.0,0.0
agent_pm_dynamic_full,v7pm_planning_memory_full,15,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.64,0.5094909090909089,3.0,0.0
agent_pm_dynamic_no_failure_memory,v7pm_no_failure_memory,5,0.3636363636363635,0.3919999999999997,0.3778181818181816,0.76,0.5176072727272726,3.0,0.0
agent_pm_dynamic_no_instability_penalty,v7pm_no_instability,5,0.3636363636363635,0.4174545454545451,0.3905454545454543,0.7,0.5155345454545452,3.0,0.0
agent_pm_dynamic_no_memory,v7pm_no_memory,5,0.3454545454545454,0.4021818181818178,0.3738181818181816,0.86,0.5359418181818181,3.0,0.0
agent_pm_dynamic_no_rarity_memory,v7pm_no_rarity_memory,5,0.3454545454545454,0.4509090909090903,0.3981818181818179,0.6199999999999999,0.5069454545454544,3.0,0.0
agent_pm_dynamic_no_utility_ema,v7pm_no_utility_ema,5,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.6399999999999999,0.5094909090909089,3.0,0.0
agent_rule_v7,v7agent_rule_early_hardquery,5,0.1999999999999998,0.5985454545454552,0.3992727272727269,0.54,0.5034327272727274,3.0,0.0
agent_rule_v7_dynamic,v7agentpm_rule_dynamic,10,0.3636363636363635,0.4363636363636359,0.3999999999999997,0.7,0.5229090909090908,3.0,0.0
hypernet_v6,hp1_baseline_hypernet_v6,5,0.0,0.588363636363637,0.2941818181818184,0.42,0.4134618181818183,3.0,0.0
```

## 6. True FiD/T5 Official Eval 结果

```csv
method,n,answer_em_mean,answer_em_std,answer_f1_mean,answer_f1_std,sp_em_mean,sp_em_std,sp_f1_mean,sp_f1_std,joint_em_mean,joint_em_std,joint_f1_mean,joint_f1_std,support_title_recall_at_k_mean,support_title_recall_at_k_std
agent_dynamic_no_bridge_guard,5,0.5762,0.0004472135954999,0.6536608747247794,8.944271910003138e-05,0.209,0.0,0.5065,0.0,0.113,1.5515838457795457e-17,0.3315276324278607,0.0,0.7445,0.0
agent_dynamic_no_hardness,5,0.577,0.0,0.6538208747247795,0.0,0.208,3.103167691559092e-17,0.506,0.0,0.113,1.5515838457795457e-17,0.3313276324278607,0.0,0.744,0.0
agent_dynamic_no_rarity,5,0.5762,0.0004472135954999,0.6536608747247794,8.944271910003138e-05,0.2088,0.0004472135954999,0.5064,0.0002236067977499,0.113,1.5515838457795457e-17,0.3314876324278607,8.944271910000657e-05,0.7444000000000001,0.00022360679775
agent_dynamic_slot,5,0.5762,0.0004472135954999,0.6536608747247794,8.944271910003138e-05,0.209,0.0,0.5065,0.0,0.113,1.5515838457795457e-17,0.3315276324278607,0.0,0.7445,0.0
agent_fixed_slot_0,5,0.5773999999999999,0.0005477225575051,0.6541700810739857,0.000182574185835,0.208,3.103167691559092e-17,0.506,0.0,0.113,1.5515838457795457e-17,0.3310022356024638,9.128709291751764e-05,0.7439,0.0002236067977499
agent_fixed_slot_1,5,0.577,0.0,0.6538208747247795,0.0,0.208,3.103167691559092e-17,0.506,0.0,0.113,1.5515838457795457e-17,0.3313276324278607,0.0,0.744,0.0
agent_fixed_slot_2,5,0.576,0.0,0.6536208747247794,0.0,0.2084,0.0005477225575051,0.5062,0.0002738612787525,0.113,1.5515838457795457e-17,0.3315276324278607,0.0,0.7445,0.0
agent_fixed_slot_3,5,0.577,0.0,0.6529208747247793,0.0,0.207,0.0,0.5045,0.0,0.114,1.5515838457795457e-17,0.3302776324278607,0.0,0.7455,0.0
agent_pm_bandit_slot,5,0.576,0.0,0.6536208747247794,0.0,0.2084,0.0005477225575051,0.5062,0.0002738612787525,0.113,1.5515838457795457e-17,0.3315276324278607,0.0,0.7445,0.0
agent_pm_dynamic_full,15,0.5761999999999999,0.0004140393356054,0.6536608747247795,8.280786712111937e-05,0.209,2.872975021810196e-17,0.5065,0.0,0.113,0.0,0.3315276324278607,0.0,0.7445000000000002,1.1491900087240784e-16
agent_pm_dynamic_no_failure_memory,5,0.5758,0.0004472135954999,0.6534208747247794,0.0004472135954999,0.208,0.001,0.5058999999999999,0.0006519202405202,0.113,1.5515838457795457e-17,0.3312476324278607,0.0005215361924162,0.7444,0.00022360679775
agent_pm_dynamic_no_instability_penalty,5,0.5762,0.0004472135954999,0.6536608747247794,8.944271910003138e-05,0.209,0.0,0.5065,0.0,0.113,1.5515838457795457e-17,0.3315276324278607,0.0,0.7445,0.0
agent_pm_dynamic_no_memory,5,0.5761999999999999,0.0004472135954999,0.653127541391446,0.0004576914697495,0.2082,0.000836660026534,0.5058999999999999,0.0006519202405202,0.113,1.5515838457795457e-17,0.330940965761194,0.0001317967009085,0.7443000000000001,0.0002738612787526
agent_pm_dynamic_no_rarity_memory,5,0.5762,0.0004472135954999,0.6536608747247794,8.944271910003138e-05,0.2088,0.0004472135954999,0.5064,0.0002236067977499,0.113,1.5515838457795457e-17,0.3314876324278607,8.944271910000657e-05,0.7444000000000001,0.00022360679775
agent_pm_dynamic_no_utility_ema,5,0.5762,0.0004472135954999,0.6536608747247794,8.944271910003138e-05,0.209,0.0,0.5065,0.0,0.113,1.5515838457795457e-17,0.3315276324278607,0.0,0.7445,0.0
agent_rule_v7,15,0.5763333333333331,0.0004879500364742,0.6536875413914461,9.759000729489676e-05,0.2086666666666667,0.0004879500364742,0.5063333333333334,0.0002439750182371,0.113,0.0,0.331460965761194,9.759000729486966e-05,0.7443333333333334,0.0002439750182371
hypernet_v6,10,0.5780000000000001,1.1702778228589004e-16,0.6539700810739857,0.0003442651863295,0.208,2.925694557147251e-17,0.506,0.0,0.113,1.4628472785736258e-17,0.3309022356024638,0.0001721325931647,0.7437,0.0002581988897471
```

## 7. Dynamic Planning Ablation

```csv
method,agent_profile,n,early_recall,bridge_recall,target_recall,diversity,hp1_score,avg_topk,budget_std
agent_dynamic_no_bridge_guard,v7pm_dynamic_no_bridge_guard,5,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.6399999999999999,0.5094909090909089,3.0,0.0
agent_dynamic_no_hardness,v7pm_dynamic_no_hardness,5,0.1999999999999998,0.5956363636363642,0.3978181818181815,0.48,0.4902981818181819,3.0,0.0
agent_dynamic_no_rarity,v7pm_dynamic_no_rarity,5,0.3454545454545454,0.4509090909090903,0.3981818181818179,0.6199999999999999,0.5069454545454544,3.0,0.0
agent_dynamic_slot,v7pm_dynamic_slot,5,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.6399999999999999,0.5094909090909089,3.0,0.0
agent_fixed_slot_0,v7pm_fixed_slot_0,5,0.0,0.6312727272727279,0.3156363636363639,0.5,0.4461963636363639,3.0,0.0
agent_fixed_slot_1,v7pm_fixed_slot_1,5,0.1999999999999998,0.5956363636363642,0.3978181818181815,0.48,0.4902981818181819,3.0,0.0
agent_fixed_slot_2,v7pm_fixed_slot_2,5,0.3999999999999997,0.399272727272727,0.3996363636363633,0.5599999999999999,0.4917163636363634,3.0,0.0
agent_fixed_slot_3,v7pm_fixed_slot_3,5,0.6000000000000006,0.1999999999999998,0.3999999999999997,0.5599999999999999,0.4760000000000001,3.0,0.0
```

## 8. Memory Ablation

```csv
method,agent_profile,n,early_recall,bridge_recall,target_recall,diversity,hp1_score,avg_topk,budget_std
agent_pm_dynamic_full,v7pm_planning_memory_full,15,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.64,0.5094909090909089,3.0,0.0
agent_pm_dynamic_no_failure_memory,v7pm_no_failure_memory,5,0.3636363636363635,0.3919999999999997,0.3778181818181816,0.76,0.5176072727272726,3.0,0.0
agent_pm_dynamic_no_instability_penalty,v7pm_no_instability,5,0.3636363636363635,0.4174545454545451,0.3905454545454543,0.7,0.5155345454545452,3.0,0.0
agent_pm_dynamic_no_memory,v7pm_no_memory,5,0.3454545454545454,0.4021818181818178,0.3738181818181816,0.86,0.5359418181818181,3.0,0.0
agent_pm_dynamic_no_rarity_memory,v7pm_no_rarity_memory,5,0.3454545454545454,0.4509090909090903,0.3981818181818179,0.6199999999999999,0.5069454545454544,3.0,0.0
agent_pm_dynamic_no_utility_ema,v7pm_no_utility_ema,5,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.6399999999999999,0.5094909090909089,3.0,0.0
```

## 9. Bandit Slot Policy

```csv
method,agent_profile,n,early_recall,bridge_recall,target_recall,diversity,hp1_score,avg_topk,budget_std
agent_pm_bandit_slot,v7pm_bandit_slot_ucb_proxy,5,0.3636363636363635,0.4290909090909086,0.3963636363636361,0.8400000000000001,0.548072727272727,3.0,0.0
agent_pm_dynamic_full,v7pm_planning_memory_full,15,0.3636363636363635,0.4327272727272723,0.3981818181818179,0.64,0.5094909090909089,3.0,0.0
agent_rule_v7_dynamic,v7agentpm_rule_dynamic,10,0.3636363636363635,0.4363636363636359,0.3999999999999997,0.7,0.5229090909090908,3.0,0.0
```

## 10. Subgroup Analysis

```csv
method,subgroup,n,early_recall,bridge_recall,support_title_recall,answer_f1,joint_f1,hp1_score,note
adaptive_v6,hard-query subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
adaptive_v6,easy-query subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
adaptive_v6,rare-domain subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
adaptive_v6,common-domain subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
adaptive_v6,hard-client subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
adaptive_v6,normal-client subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
adaptive_v6,early-evidence-needed subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
adaptive_v6,bridge-heavy subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,hard-query subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,easy-query subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,rare-domain subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,common-domain subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,hard-client subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,normal-client subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,early-evidence-needed subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_bridge_guard,bridge-heavy subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,hard-query subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,easy-query subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,rare-domain subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,common-domain subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,hard-client subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,normal-client subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,early-evidence-needed subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_hardness,bridge-heavy subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,hard-query subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,easy-query subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,rare-domain subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,common-domain subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,hard-client subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,normal-client subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,early-evidence-needed subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_no_rarity,bridge-heavy subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,hard-query subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,easy-query subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,rare-domain subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,common-domain subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,hard-client subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,normal-client subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,early-evidence-needed subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_dynamic_slot,bridge-heavy subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,hard-query subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,easy-query subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,rare-domain subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,common-domain subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,hard-client subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,normal-client subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,early-evidence-needed subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_0,bridge-heavy subset,275,0.0,0.6312727272727279,,,,0.4461963636363638,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,hard-query subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,easy-query subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,rare-domain subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,common-domain subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,hard-client subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,normal-client subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,early-evidence-needed subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_1,bridge-heavy subset,275,0.1999999999999998,0.5956363636363642,,,,0.4902981818181819,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,hard-query subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,easy-query subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,rare-domain subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,common-domain subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,hard-client subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,normal-client subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,early-evidence-needed subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_2,bridge-heavy subset,275,0.3999999999999997,0.399272727272727,,,,0.4917163636363634,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,hard-query subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,easy-query subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,rare-domain subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,common-domain subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,hard-client subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,normal-client subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,early-evidence-needed subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_fixed_slot_3,bridge-heavy subset,275,0.6000000000000006,0.1999999999999998,,,,0.4760000000000001,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,hard-query subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,easy-query subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,rare-domain subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,common-domain subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,hard-client subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,normal-client subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,early-evidence-needed subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_bandit_slot,bridge-heavy subset,275,0.3636363636363635,0.4290909090909086,,,,0.548072727272727,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,hard-query subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,easy-query subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,rare-domain subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,common-domain subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,hard-client subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,normal-client subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,early-evidence-needed subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_full,bridge-heavy subset,825,0.3636363636363634,0.4327272727272723,,,,0.5094909090909089,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,hard-query subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,easy-query subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,rare-domain subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,common-domain subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,hard-client subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,normal-client subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,early-evidence-needed subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_failure_memory,bridge-heavy subset,275,0.3636363636363635,0.3919999999999997,,,,0.5176072727272726,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,hard-query subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,easy-query subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,rare-domain subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,common-domain subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,hard-client subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,normal-client subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,early-evidence-needed subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_instability_penalty,bridge-heavy subset,275,0.3636363636363635,0.4174545454545451,,,,0.5155345454545452,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,hard-query subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,easy-query subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,rare-domain subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,common-domain subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,hard-client subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,normal-client subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,early-evidence-needed subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_memory,bridge-heavy subset,275,0.3454545454545454,0.4021818181818178,,,,0.5359418181818181,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,hard-query subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,easy-query subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,rare-domain subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,common-domain subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,hard-client subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,normal-client subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,early-evidence-needed subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_rarity_memory,bridge-heavy subset,275,0.3454545454545454,0.4509090909090903,,,,0.5069454545454544,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,hard-query subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,easy-query subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,rare-domain subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,common-domain subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,hard-client subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,normal-client subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,early-evidence-needed subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_pm_dynamic_no_utility_ema,bridge-heavy subset,275,0.3636363636363635,0.4327272727272723,,,,0.509490909090909,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,hard-query subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,easy-query subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,rare-domain subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,common-domain subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,hard-client subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,normal-client subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,early-evidence-needed subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7,bridge-heavy subset,275,0.1999999999999998,0.5985454545454552,,,,0.5034327272727274,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,hard-query subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,easy-query subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,rare-domain subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,common-domain subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,hard-client subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,normal-client subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,early-evidence-needed subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
agent_rule_v7_dynamic,bridge-heavy subset,550,0.3636363636363634,0.4363636363636359,,,,0.5229090909090907,proxy strict subgroup; query-level merge after official eval
hypernet_v6,hard-query subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
hypernet_v6,easy-query subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
hypernet_v6,rare-domain subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
hypernet_v6,common-domain subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
hypernet_v6,hard-client subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
hypernet_v6,normal-client subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
hypernet_v6,early-evidence-needed subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
hypernet_v6,bridge-heavy subset,275,0.0,0.588363636363637,,,,0.4134618181818183,proxy strict subgroup; query-level merge after official eval
```

## 11. Per-query Behavior Case Study

见 `analysis/per_query_behavior.csv` 与 `analysis/representative_cases.md`。

## 12. 统计检验

```csv
comparison,metric,n,mean_delta,wilcoxon_stat,p_value
agent_rule_v7_dynamic vs agent_pm_dynamic_full,hp1_multihop_score,30,0.0134181818181818,0.0,1.584089649817868e-05
agent_rule_v7_dynamic vs agent_pm_bandit_slot,hp1_multihop_score,10,-0.0251636363636363,0.0,0.001953125
agent_pm_dynamic_full vs agent_pm_dynamic_no_memory,hp1_multihop_score,15,-0.0264509090909091,6.0,0.0020739283941541
agent_pm_dynamic_full vs agent_pm_dynamic_no_failure_memory,hp1_multihop_score,15,-0.0081163636363637,33.0,0.1236290924103736
agent_dynamic_slot vs agent_fixed_slot_1,hp1_multihop_score,5,0.0191927272727269,0.0,0.0625
agent_dynamic_slot vs agent_fixed_slot_2,hp1_multihop_score,5,0.0177745454545454,0.0,0.0625
```

## 13. 失败与限制

若 FiD/T5 reader fallback、OOM、缺失 run 或 official baseline 不完整，必须在本节记录。当前脚本已先修复 sentencepiece 并强制 smoke check T5 tokenizer。

## 14. 下一步建议

优先比较 `agent_rule_v7_dynamic` 与 `agent_pm_dynamic_full` 的 strict 和 true FiD/T5 指标；若整体差异小，聚焦 hard-query、rare-domain、hard-client 子集。

## 15. 可写入论文的结论段

V7-agent-PM tests whether planning-memory upload selection can improve same-budget federated RAG beyond dynamic early-slot heuristics. The final claim should depend on true FiD/T5 and subgroup evidence rather than fallback reader scores.
