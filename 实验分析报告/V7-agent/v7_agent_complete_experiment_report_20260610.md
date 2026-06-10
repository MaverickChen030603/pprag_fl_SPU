# V7-agent 完整实验报告

生成时间：2026-06-10T16:00:00

## 0. 执行摘要

V7-agent 是从 V7-HP1 平行独立出的 agent 主线实验目录，目标是在不扩大通信预算的条件下，让 `agent_rule_v7` 对 HotpotQA 多跳证据链更敏感。核心改动是把 hard-query focused policy 与 early-evidence coverage 结合：在固定 top-k=3 的上传预算内，保留 high-layer bridge blocks，同时稳定插入一个 low-layer early evidence block。

主要结果：

- Upstream 训练：`hp1_budget_aligned` 共 12/12 runs 完成。
- Strict diagnostic：所有方法 `avg_budget_topk_hp1 = 3.0000 ± 0.0000`，same-budget 成立。
- `agent_rule_v7` 将 `early_evidence_recall_hp1` 从 0.0000 提升到 0.2000。
- `agent_rule_v7` 将 `hp1_multihop_score` 从 0.4145 提升到 0.5084，相对最佳 baseline +0.0939，约 +22.6%。
- Official-style Hotpot eval：answer/joint F1 暂未出现明显传导，各方法 `joint_f1 = 0.4400`；`agent_rule_v7` 仅在 support title recall 上有很小提升 0.7350 -> 0.7375。

结论边界：V7-agent 已证明 agent policy 能在 same-budget 下显著改变 block selection，并捕获 early evidence；但尚不能宣称最终 Hotpot QA F1/EM 已显著提升。

## 1. 项目背景与实验目的

V6/V7 早期结果显示，在严格 Same-Budget Protocol 下，常规选择策略容易集中于 high-layer bridge/semantic blocks，对 HotpotQA 这类多跳任务中的低层 early evidence 表征不敏感。V7-agent 的目标不是扩大 payload，而是验证 Client Agent 是否能通过局部记忆、hard-query 对齐和显式 evidence coverage，在相同 top-k 预算内改变上传 block 的结构。

研究问题：

1. 在 `top-k=3` same-budget 下，agent policy 是否能让 early-evidence blocks 进入上传集合？
2. 这种选择行为改变是否带来 HP1 strict multihop diagnostic 的正信号？
3. 该正信号是否已经传导到 official-style Hotpot answer/supporting-fact/joint metrics？

## 2. 实验目录与代码状态

- 项目目录：`/home/iiserver31/projects/FedE4RAG-main/V7-agent`
- 训练输出：`V7-agent/outputs/pprag_fl_v7_agent/hp1_budget_aligned`
- strict eval：`V7-agent/outputs/hp1_strict_eval/hp1_budget_aligned`
- official eval：`V7-agent/outputs/hotpot_official_eval/hp1_budget_aligned`
- 报告归档：`实验分析报告/V7-agent/archive_v7_agent_budget_aligned_20260610`

关键代码/脚本：

- `V7-agent/agent_core.py`: AgentScorer、early_evidence_alignment、hard_query_focused 权重。
- `V7-agent/upload_selectors.py`: same-budget top-k 内 early evidence coverage 插入逻辑。
- `V7-agent/run_experiment_suite.py`: `agent_rule_v7` 主线 profile `v7agent_rule_early_hardquery`。
- `run_v7_agent_all.sh`: upstream + strict eval 自动执行脚本。
- `run_v7_agent_official_eval.sh`: official-style Hotpot eval 批处理脚本。

## 3. 方法设计

V7-agent 的主线方法为 `agent_rule_v7`，采用 `hard_query_focused` 策略模式。它只改变 block ranking，不改变上传预算。具体机制：

- Local utility：保留来自 hypernet/value-density 的本地重要性。
- Memory utility：使用历史选择/utility EMA 作为噪声过滤。
- Hard-query alignment：困难查询场景下提高与下游失败相关 block 的分数。
- Early-evidence prior：显式定义 `embeddings` 与 `encoder.layer.0-3` 为 HP1 early-evidence blocks。
- Coverage constraint：当 top-k 预览中没有 early block 时，用最高分 early block 替换 top-k 末位；最终仍严格保留 `top-k=3`。
- Instability penalty 调整：hard-query 模式下降低 instability penalty，避免 high-layer bridge blocks 被过度惩罚。

该设计的意图是形成 `2 bridge + 1 early evidence` 的上传结构，而不是把预算扩展到 4 或牺牲所有 high-layer bridge blocks。

## 4. 实验设置

- 数据集：HotpotQA fullwiki 派生 `FedE/select_data_hotpot_train_5000.json`。
- Suite：`hp1_budget_aligned`。
- Methods：`hypernet_v6`, `adaptive_v6`, `agent_bandit_v7`, `agent_rule_v7`。
- Seeds：0, 1, 2。
- Clients：5。
- Rounds：12。
- Top-k budget：3。
- Warmup：1 round。
- Strict metrics：bridge/early/target recall、selection diversity、HP1 multihop diagnostic score。
- Official-style eval：200 compact Hotpot examples per run，retriever sentence ranking + heuristic answer。

## 5. 完成情况

- Upstream final artifacts：12/12
- Strict eval records：12/12
- Official eval records：12/12
- 当前无 V7-agent 运行进程。

## 6. Strict Diagnostic 主结果

| method | n | top-k | bridge recall | early recall | target recall | diversity | HP1 score |
|---|---:|---:|---:|---:|---:|---:|---:|
| hypernet_v6 | 3 | 3.0000 | 0.5842 | 0.0000 | 0.2921 | 0.4333 | 0.4145 |
| adaptive_v6 | 3 | 3.0000 | 0.5842 | 0.0000 | 0.2921 | 0.4333 | 0.4145 |
| agent_bandit_v7 | 3 | 3.0000 | 0.6048 | 0.0000 | 0.3024 | 0.5000 | 0.4359 |
| agent_rule_v7 | 3 | 3.0000 | 0.5976 | 0.2000 | 0.3988 | 0.5667 | 0.5084 |

核心解读：

- Same-budget 成立：四类方法 top-k 均为 3.0000。
- `agent_rule_v7` early recall：0.0000 -> 0.2000。
- `agent_rule_v7` target recall：0.2921 -> 0.3988，相对提升 36.5%。
- `agent_rule_v7` HP1 score：0.4145 -> 0.5084，相对提升 22.6%。
- `agent_bandit_v7` 有小幅 HP1 score 正信号，但 early recall 仍为 0，因此不是当前论文主线。

## 7. 逐 Seed 稳定性

| seed | hypernet score | agent_rule score | score gap | agent_rule early |
|---:|---:|---:|---:|---:|
| 0 | 0.4069 | 0.5146 | +0.1077 | 0.2000 |
| 1 | 0.4041 | 0.5146 | +0.1105 | 0.2000 |
| 2 | 0.4326 | 0.4960 | +0.0634 | 0.2000 |

三个 seed 上 `agent_rule_v7` 均保持正 gap，并且 early recall 稳定为 0.2000，说明这不是单 seed 偶然波动。

## 8. 选择行为分析

| method | events | early share | bridge share | other share | top selected blocks |
|---|---:|---:|---:|---:|---|
| hypernet_v6 | 165 | 0.0% | 77.6% | 22.4% | pooler:165, encoder.layer.8:160, encoder.layer.11:153, encoder.layer.7:139, encoder.layer.9:4 |
| adaptive_v6 | 165 | 0.0% | 77.6% | 22.4% | pooler:165, encoder.layer.8:160, encoder.layer.11:153, encoder.layer.7:139, encoder.layer.9:4 |
| agent_bandit_v7 | 165 | 0.0% | 76.9% | 23.1% | pooler:165, encoder.layer.8:163, encoder.layer.11:159, encoder.layer.7:149, encoder.layer.9:12 |
| agent_rule_v7 | 165 | 25.0% | 74.7% | 0.3% | encoder.layer.8:165, pooler:165, encoder.layer.11:163, encoder.layer.0:154, encoder.layer.3:11 |

选择行为解释：baseline/adaptive/bandit 的 early share 均为 0；`agent_rule_v7` 的 early share 为 25%，对应 top-k=3 中每次稳定纳入一个 early-evidence block，同时 bridge share 仍接近 75%。这正是 V7-agent policy 的目标结构。

## 9. Official-Style Hotpot Eval

| method | n | answer EM | answer F1 | sp EM | sp F1 | joint EM | joint F1 | support title recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hypernet_v6 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7350 |
| adaptive_v6 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7350 |
| agent_bandit_v7 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7350 |
| agent_rule_v7 | 3 | 0.8300 | 0.8310 | 0.2000 | 0.5050 | 0.1850 | 0.4400 | 0.7375 |

Official-style 解读：

- `agent_rule_v7` strict gain 未明显传导到 answer/joint F1：best baseline joint F1 = 0.4400，`agent_rule_v7` joint F1 = 0.4400。
- `agent_rule_v7` support title recall 有很小提升：0.7350 -> 0.7375。
- 当前 evaluator 是 retriever sentence ranking + heuristic answer，不是端到端生成式 reader，因此它可能对 block-level early evidence 改善不够敏感。

## 10. 论文可用结论

可以写入论文的主结论：

> Under an identical Top-K communication budget, the rule-based client agent changes the upload structure toward multihop evidence coverage. It recovers early-evidence block recall from 0.0000 to 0.2000 while preserving bridge-block recall, yielding a +22.6% relative gain in the HP1 multihop diagnostic score over the strongest V6 baseline.

中文版本：

> 在严格相同通信预算下，`agent_rule_v7` 并未依赖额外 payload，而是通过 hard-query 对齐和 early-evidence coverage 改变上传 block 结构，使 HotpotQA 多跳证据链中的低层证据块稳定进入 top-k 上传集合，从而显著提升 HP1 strict multihop diagnostic score。

不应过度声称的部分：

- 不能说 V7-agent 已经显著提升最终 Hotpot QA F1/EM。
- 不能把 strict diagnostic score 等同于 official supporting-fact F1。
- 当前 official-style eval 持平，说明 selection gain 到 downstream QA 的传导仍需更强 reader/reranker 验证。

## 11. 局限与风险

- 样本规模：strict 只有 3 seeds，official eval 每 run 200 compact examples。
- Evaluator 限制：answer 由启发式规则产生，不能代表完整 RAG generation。
- 数据模式：official-style eval 使用 compact Hotpot 派生数据，不是完全官方 fullwiki dev pipeline。
- 方法风险：early evidence coverage 是显式 HP1 inductive bias，需要在更多任务上验证泛化。
- Agent-bandit 尚未激活 early evidence；若论文需要 bandit 主线，还需单独优化。

## 12. 下一步建议

1. 跑 `V7-agent hard/rare/full`，确认 early-evidence gain 在 hard-query 与 rare-domain 场景是否更强。
2. 接一个真实 reader/generator，比如 FiD/T5/Gemma 或现有 RAGTest 生成式评估，而不是 heuristic answer。
3. 增加 ablation：去掉 early prior、去掉 coverage replacement、去掉 memory，证明是哪一项带来 early recall。
4. 将 official eval 扩到 1000 examples，并优先看 support-title recall、supporting-fact F1、joint F1。
5. 优化 `agent_bandit_v7`，把 early-evidence reward 纳入 bandit exploration/exploitation。

## 13. 文件索引

- Strict CSV：`V7-agent/outputs/hp1_strict_eval/hp1_budget_aligned/hp1_strict_summary.csv`
- Official eval CSV：`实验分析报告/V7-agent/archive_v7_agent_budget_aligned_20260610/hotpot_official_eval_summary_200.csv`
- 自动报告：`实验分析报告/V7-agent/v7_agent_auto_analysis_20260610_143930.md`
- 论文分析：`实验分析报告/V7-agent/v7_agent_result_interpretation_and_paper_analysis_latest.md`
- 本完整报告：`实验分析报告/V7-agent/v7_agent_complete_experiment_report_20260610.md`
