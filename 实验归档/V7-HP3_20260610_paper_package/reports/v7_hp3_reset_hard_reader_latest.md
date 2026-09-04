# V7-HP3 Reset Hard-Reader 实验报告

生成时间: 2026-06-09T20:05:45

## 实验目的

HP3 是对 HP2 打平结果的三步重置：

- Reader 从 T5-small 提升到强 reader，默认 `google/flan-t5-large`，脚本也支持 Qwen/Llama causal reader。
- Reader-aware reward 改为阶梯式 high-contrast：top block +10，bottom block -5，中间 0。
- 从 HP2 per-query 成绩单反筛 Recoverable-Hard 100，用于 hard-case official 与 reader 分析。

## Hard100 筛选诊断

- 样本数: 100
- 严格 recoverable 数: 1
- 选入严格 recoverable 数: 1
- fallback medium-hard 数: 178
- easy removed: 7
- impossible removed: 46

注意：严格 recoverable 若很少，说明 HP2 各方法 per-query 输出同质化；HP3 的主要检验点变成强 reader 与 step reward 能否在 hard100 上制造新的分离。

## Official Hard100

| method | profile | runs | answer_access_at_k | support_title_recall_at_k | sp_f1 | joint_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| agent_memory_v7hp2 | hp2_reader_memory_agent | 3 | 0.2200 | 0.4400 | 0.3374 | 0.0021 |
| agent_tail_v7hp2 | hp2_reader_tail_agent | 3 | 0.2200 | 0.4433 | 0.3397 | 0.0021 |
| hypernet_v6 | hp2_baseline_adaptive_v6 | 3 | 0.2200 | 0.4400 | 0.3374 | 0.0021 |
| hypernet_v6 | hp2_baseline_hypernet_v6 | 3 | 0.2200 | 0.4400 | 0.3374 | 0.0021 |


## Strong Reader Hard100

| method | profile | runs | answer_em | answer_f1 | sp_f1 | joint_f1 | answer_access_at_k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agent_memory_v7hp2 | hp2_reader_memory_agent | 3 | 0.4900 | 0.5180 | 0.3374 | 0.1656 | 0.2200 |
| agent_tail_v7hp2 | hp2_reader_tail_agent | 3 | 0.4900 | 0.5180 | 0.3397 | 0.1656 | 0.2200 |
| hypernet_v6 | hp2_baseline_adaptive_v6 | 3 | 0.4900 | 0.5180 | 0.3374 | 0.1656 | 0.2200 |
| hypernet_v6 | hp2_baseline_hypernet_v6 | 3 | 0.4900 | 0.5180 | 0.3374 | 0.1656 | 0.2200 |


## 初步判断

- Official agent-baseline best joint_f1 gap: +0.0000
- Reader agent-baseline best joint_f1 gap: +0.0000

若 gap 为正且 reader-aware memory/tail 同时提高 support 与 answer F1，可作为 V7 agent 正信号；若仍打平，则说明 block-selection 对 retriever 表征影响被当前训练/评估链路抹平，需要进入更强训练分叉或真实 online reader reward。
