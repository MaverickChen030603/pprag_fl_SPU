# V20 Stage R4 冻结双 Reader 端到端实验完整进展报告

**项目：** ARC-FedSearch / ProbeRoute-FedRAG  
**阶段：** R4 Frozen Dual-Reader End-to-End Evaluation  
**报告日期：** 2026-08-12  
**实验状态：** `probe_route_end_to_end_confirmed`  
**当前决策：** R4 通过，冻结的 ProbeRoute 检索信号已稳定转化为跨数据集、跨 Reader 的端到端 Joint F1 收益；final test 继续封存，进入论文整理与一次性 R5 决策阶段。  
**代码版本：** GitHub `main`，提交 `13091c6`（修正 R4 bootstrap p-value 实现）。

---

## 1. 执行摘要

V20 前序检索实验已经证明，在联邦多跳检索中，瓶颈并不只是单客户端局部排序，还包括客户端资源选择、有限通信预算下的候选压缩，以及不同客户端得分不可直接比较的问题。R3 最终形成两条冻结路线：无需监督的 `label_free_proberoute`，以及使用轻量 logistic probe-ranker 的 `logistic_proberoute`。R4 的目标不是继续调检索，而是回答最关键的论文问题：**检索层的 complete-support 改善，是否能在不修改上下文、prompt、reader 或 decoder 的条件下转化为真实问答收益。**

本轮在 HotpotQA、2WikiMultiHopQA、MuSiQue 三个互相补充的多跳数据集上，各使用冻结的 300-query holdout；比较 federated baseline、label-free ProbeRoute、logistic ProbeRoute 和 centralized retrieval reference；使用 FLAN-T5-Large 与 UnifiedQA-T5-Large 两个冻结 Reader。共完成：

- 3 个数据集；
- 4 种上下文方法；
- 每个数据集 300 条查询；
- 2 个 Reader；
- 3,600 个冻结 method-query contexts；
- 7,200 条正式 Reader 预测记录；
- 24 个 dataset-reader-method 聚合结果；
- 6 个预注册 primary dataset-reader cells。

核心结果如下：

1. `logistic_proberoute` 相对 federated baseline 的 Joint F1 在 6/6 个 dataset-reader cells 中均为正，95% paired-bootstrap CI 下界全部大于 0。
2. Joint F1 绝对提升范围为 **+1.77 至 +6.32 个百分点**；六个 cell 的宏平均提升为 **+3.87 个百分点**。
3. SP F1 在 6/6 个 cells 中显著提高，绝对提升范围为 **+4.66 至 +11.03 个百分点**，宏平均为 **+7.51 个百分点**。
4. Answer F1 在 6/6 个 cells 中方向为正，宏平均为 **+3.77 个百分点**；其中 FLAN 三个数据集和 HotpotQA UnifiedQA 通过 BH-FDR，2Wiki 与 MuSiQue 的 UnifiedQA Answer F1 单项仍属方向性结果。
5. support-rescue 子集上，Joint F1 提升远大于总体均值，说明主收益确实通过“补齐多跳证据”传递，而不是由无关上下文扰动偶然产生。
6. HotpotQA 上 ProbeRoute 的 Joint/SP 指标超过同合同 centralized reference；这说明 centralized reference 只能作为相同检索合同下的参考点，不能称为数学上界。
7. learned logistic 相对 label-free 的额外 Joint F1 增益较小且多数 CI 跨零。当前最稳妥的论文结论是 **ProbeRoute 方法族有效，logistic 版本提供一致的数值最优或近优表现，但其相对 label-free 的独立增量尚未全面确认。**

---

## 2. 研究问题与可验证假设

### 2.1 研究问题

在固定通信预算和冻结本地检索器的联邦多跳 RAG 中，server 不能直接看到所有客户端文档，只能先选择少量客户端，再接收每个客户端返回的少量候选。传统 centroid/topic routing 和 raw-score merge 容易出现三类损失：

1. 正确 evidence 所在客户端未被选择；
2. 正确客户端被选择，但局部候选未进入传输预算；
3. 多客户端候选已传输，但异构得分尺度导致关键证据在全局合并时被压低。

V20 的核心思想是让客户端发送紧凑 query-conditioned probe signals，由 server 在不读取文档正文、不过度增加通信的情况下改善资源选择。R4 进一步检验，这种 retrieval-side 选择是否真正有利于下游 reader，而不只是让 support proxy 指标变好。

### 2.2 预注册主假设

在完全冻结的检索和 Reader 合同下，`logistic_proberoute` 相对 `federated_baseline` 应当：

- 提高 query-level Joint F1；
- 至少在两个数据集上保持正向；
- 不造成任何 dataset-reader cell 的总体 Answer F1 下降；
- retrieval support rescue 应伴随正向 Reader Joint F1 变化；
- 结果不能依赖 final-test 标签、gold routing 特征或 reader 后验调参。

### 2.3 本轮不检验的内容

R4 不重新训练 router、local ranker、merge calibrator 或 Reader，也不调 context K、prompt、beam、温度、阈值和通信预算。它是一次冻结的端到端转化验证，不是新的模型搜索阶段。

---

## 3. 实验配置

### 3.1 数据集与样本

| 数据集 | 冻结样本 | 作用 |
|---|---:|---|
| HotpotQA | 300 | 主数据集；验证 bridge/comparison 型多跳推理 |
| 2WikiMultiHopQA | 300 | 跨数据集迁移；实体链和组合关系较强 |
| MuSiQue | 300 | 更复杂的组合式多跳问题，检验泛化难度 |

三个数据集均沿用 R3 冻结 holdout query IDs，不根据 R4 Reader 结果重新采样。final test 未访问。

### 3.2 比较方法

| 方法 | 角色 | 说明 |
|---|---|---|
| `federated_baseline` | Primary control | 冻结的 inherited federated route；Hotpot 使用 H0 合同 |
| `label_free_proberoute` | 无监督方法 | 使用冻结 query-conditioned probe 特征与无标签选择规则 |
| `logistic_proberoute` | 主方法 | 使用 Probe-Train 学得的轻量 logistic client ranker，holdout 上不调参 |
| `centralized_retrieval_reference` | 参考点 | 在同一 query IDs 上重放冻结 V17 centralized hybrid retrieval；不是上界 |

HotpotQA 的 C0 static Top-3 只用于前序成本/Pareto 比较，不进入 R4 Reader 主表。主 federated baseline 始终是 H0 inherited route。

### 3.3 冻结检索与上下文合同

- 客户端接触预算：`Bc = 3`；
- local materialization depth：10；
- 每客户端传输 5 个文档；
- 总传输预算：15 文档；
- server retrieval pool：raw merged Top-10；
- Reader context：冻结顺序中的前 5 个文档；
- 不补齐、不重复 doc ID，不做 Reader-aware rerank；
- centralized comparator 使用相同 Reader context K 和同一评估程序。

### 3.4 Reader 合同

| Reader | 冻结 checkpoint revision |
|---|---|
| `google/flan-t5-large` | `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` |
| `allenai/unifiedqa-v2-t5-large-1363200` | `1d3b8e13b29dbd161494b0b15428378f4713c418` |

共同推理设置：

- context serialization 沿用冻结 V17 prompt；
- context 最大 4,000 characters；
- tokenizer 输入最大 1,024 tokens，右截断；
- greedy decoding；
- `num_beams=1`；
- `do_sample=False`；
- `max_new_tokens=32`；
- CUDA float16，batch size 4；
- support extraction 使用冻结 V16 support predictor 与 deterministic top-two fallback。

### 3.5 指标与统计

- Answer EM / F1；
- Supporting-fact EM / F1；
- Joint EM / F1；
- Reader context complete-support rate；
- query-level paired win/tie/loss；
- 5,000 次 query-level paired bootstrap；
- primary 指标为 logistic ProbeRoute 对 federated baseline 的 Joint F1；
- secondary Answer/SP 检验使用 BH-FDR。

统计实现已在本轮收尾时审计：原脚本的 `two_sided_p` 使用带 tie 的样本符号比例，可能产生大于 1 的非法值。现已改为基于 5,000 次 bootstrap mean 分布的双侧尾概率，并使用 plus-one correction，保证 p-value 位于 `[0,1]`。该修复不改变预测、均值、CI 或 W/T/L，只修正 p-value 与 BH-FDR。修复提交为 `13091c6`。

---

## 4. 主结果

### 4.1 FLAN-T5-Large

| 数据集 | 方法 | Answer F1 | SP F1 | Joint F1 | Complete Support |
|---|---|---:|---:|---:|---:|
| 2Wiki | Federated baseline | 0.3951 | 0.2664 | 0.1497 | 0.1800 |
| 2Wiki | Label-free ProbeRoute | 0.4301 | 0.3652 | 0.2045 | 0.2400 |
| 2Wiki | Logistic ProbeRoute | **0.4387** | **0.3767** | **0.2129** | **0.2567** |
| 2Wiki | Centralized reference | 0.4602 | 0.4154 | 0.2499 | 0.3333 |
| HotpotQA | Federated baseline | 0.5444 | 0.3529 | 0.2319 | 0.4333 |
| HotpotQA | Label-free ProbeRoute | 0.5831 | 0.3986 | 0.2628 | 0.5333 |
| HotpotQA | Logistic ProbeRoute | **0.5964** | **0.3995** | **0.2693** | **0.5567** |
| HotpotQA | Centralized reference | 0.5948 | 0.3839 | 0.2603 | 0.5600 |
| MuSiQue | Federated baseline | 0.2209 | 0.3105 | 0.0859 | 0.1133 |
| MuSiQue | Label-free ProbeRoute | 0.2515 | 0.3729 | 0.1068 | 0.1600 |
| MuSiQue | Logistic ProbeRoute | **0.2554** | **0.3788** | **0.1120** | **0.1633** |
| MuSiQue | Centralized reference | 0.2655 | 0.4152 | 0.1268 | 0.2200 |

### 4.2 UnifiedQA-T5-Large

| 数据集 | 方法 | Answer F1 | SP F1 | Joint F1 | Complete Support |
|---|---|---:|---:|---:|---:|
| 2Wiki | Federated baseline | 0.3662 | 0.2664 | 0.1336 | 0.1800 |
| 2Wiki | Label-free ProbeRoute | **0.3927** | 0.3652 | 0.1859 | 0.2400 |
| 2Wiki | Logistic ProbeRoute | 0.3865 | **0.3767** | **0.1883** | **0.2567** |
| 2Wiki | Centralized reference | 0.3906 | 0.4154 | 0.2100 | 0.3333 |
| HotpotQA | Federated baseline | 0.4992 | 0.3529 | 0.2162 | 0.4333 |
| HotpotQA | Label-free ProbeRoute | 0.5531 | 0.3986 | 0.2470 | 0.5333 |
| HotpotQA | Logistic ProbeRoute | **0.5559** | **0.3995** | **0.2491** | **0.5567** |
| HotpotQA | Centralized reference | 0.5403 | 0.3839 | 0.2429 | 0.5600 |
| MuSiQue | Federated baseline | 0.2114 | 0.3105 | 0.0799 | 0.1133 |
| MuSiQue | Label-free ProbeRoute | 0.2286 | 0.3729 | 0.0937 | 0.1600 |
| MuSiQue | Logistic ProbeRoute | **0.2305** | **0.3788** | **0.0976** | **0.1633** |
| MuSiQue | Centralized reference | 0.2387 | 0.4152 | 0.1137 | 0.2200 |

### 4.3 Primary paired effects：Logistic ProbeRoute vs Federated Baseline

| 数据集 | Reader | Δ Answer F1 | Δ SP F1 | Δ Joint F1 | Joint 95% CI | p | Joint W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | FLAN | +0.0436 | +0.1103 | **+0.0632** | [+0.0404, +0.0862] | 0.0004 | 47/247/6 |
| 2Wiki | UnifiedQA | +0.0203 | +0.1103 | **+0.0547** | [+0.0331, +0.0773] | 0.0004 | 47/245/8 |
| HotpotQA | FLAN | +0.0520 | +0.0466 | **+0.0374** | [+0.0205, +0.0551] | 0.0004 | 33/258/9 |
| HotpotQA | UnifiedQA | +0.0567 | +0.0466 | **+0.0329** | [+0.0168, +0.0494] | 0.0004 | 32/254/14 |
| MuSiQue | FLAN | +0.0344 | +0.0683 | **+0.0261** | [+0.0132, +0.0410] | 0.0004 | 21/273/6 |
| MuSiQue | UnifiedQA | +0.0190 | +0.0683 | **+0.0177** | [+0.0056, +0.0309] | 0.0012 | 15/280/5 |

这里的 p 是 bootstrap 双侧尾概率并带 plus-one correction；0.0004 是 5,000 次 resampling 下的最小可报告非零值。Joint F1 为预注册 primary，不做 FDR；Answer/SP 为 secondary 并报告 BH-FDR。

### 4.4 宏平均效果

将六个 dataset-reader cells 等权平均：

| 方法 vs baseline | Δ Answer F1 | Δ SP F1 | Δ Joint F1 |
|---|---:|---:|---:|
| Label-free ProbeRoute | +0.0336 | +0.0690 | +0.0339 |
| Logistic ProbeRoute | **+0.0377** | **+0.0751** | **+0.0387** |

宏平均只用于总结跨 cell 的量级，不代替 query-level 配对统计，也不应作为单独显著性检验。

---

## 5. 结果解读

### 5.1 最主要结论：retrieval signal 已转化为真实 QA signal

六个 primary cells 的 Joint F1 全部为正且 CI 不跨零，排除了“只改善 retrieval complete-support、Reader 完全不受益”的失败情形。效果在两个 Reader 上方向一致，说明收益不是 FLAN 特有的 prompt 偏好，也不是单个 checkpoint 的偶然表现。

### 5.2 SP 增益最稳定，Answer 增益较依赖 Reader

Logistic ProbeRoute 的 SP F1 在所有数据集和 Reader 上均显著提高，BH-FDR q 约为 0.001；这与方法目标一致：它首先改善跨客户端多跳证据的共同可达性。

Answer F1 六个 cell 都为正，但强度不同：

- 2Wiki FLAN：+0.0436，q=0.0064；
- Hotpot FLAN：+0.0520，q=0.0010；
- Hotpot UnifiedQA：+0.0567，q=0.0010；
- MuSiQue FLAN：+0.0344，q=0.0042；
- 2Wiki UnifiedQA：+0.0203，q=0.2909；
- MuSiQue UnifiedQA：+0.0190，q=0.1204。

因此，论文中可以稳健声称“Answer F1 无总体下降且宏平均提高”，但不能声称每个数据集和 Reader 的 Answer F1 都达到统计显著。

### 5.3 Label-free ProbeRoute 本身已形成强基线

Label-free ProbeRoute 相对 federated baseline 的 Joint F1 在六个 cells 中也全部为正，宏平均 +0.0339。其意义是：改善 resource selection 不一定依赖复杂学习器，紧凑 probe features 与合理的无标签选择规则已经能解决相当部分路由损失。

### 5.4 Logistic 的增量价值存在，但不宜夸大

Logistic ProbeRoute 的六-cell Joint F1 宏平均比 label-free 高约 0.0047，但逐 cell 的 logistic-vs-label-free Joint CI 大多跨零。2Wiki 的 SP F1 增量约 +0.0115 且通过 FDR；其他数据集的额外优势较弱。

论文主线应按以下强度排序：

1. **强结论：** ProbeRoute family 相对 inherited federated routing 在 retrieval 和 reader outcomes 上均稳定改善。
2. **中等结论：** learned logistic variant 提供一致的整体数值优势和最佳平均 operating point。
3. **暂不主张：** logistic learning 在所有数据集和 Reader 上显著优于 label-free rule。

---

## 6. 机制分析

### 6.1 Complete-support 改善

Logistic ProbeRoute 的 Reader context complete-support rate：

| 数据集 | Baseline | Logistic | 绝对增益 | Rescue / Harm queries |
|---|---:|---:|---:|---:|
| 2Wiki | 0.1800 | 0.2567 | +0.0767 | 24 / 1 |
| HotpotQA | 0.4333 | 0.5567 | +0.1234 | 39 / 2 |
| MuSiQue | 0.1133 | 0.1633 | +0.0500 | 16 / 1 |

三个数据集均满足 rescue 明显大于 harm。该结果直接连接了 R3 的 retrieval success 与 R4 的 Reader 结果。

### 6.2 Support rescue 对 Reader 的条件收益

在从 incomplete support 变为 complete support 的 T1 rescue queries 上：

| 数据集 | N | FLAN Δ Joint F1 | UnifiedQA Δ Joint F1 |
|---|---:|---:|---:|
| 2Wiki | 24 | +0.4266 | +0.3155 |
| HotpotQA | 39 | +0.2682 | +0.2460 |
| MuSiQue | 16 | +0.3615 | +0.2625 |

这些条件效应远大于全样本均值，表明端到端收益集中发生在机制预期的 evidence-rescue queries 上。它不是可部署选择条件，也不能当作总体效果；其作用是做机制归因。

### 6.3 Harm 数量与剩余风险

Support 从 complete 变为 incomplete 的 T3 harm 很少：2Wiki 1 条、HotpotQA 2 条、MuSiQue 1 条。Hotpot 的两个 harm query 在 FLAN 上平均 Joint F1 下降约 0.0879；说明路由策略虽总体显著为正，仍不能被描述为逐 query 单调安全。

此外，Hotpot complete-support 已保持的 T2 子集上存在轻微平均 Joint 下降，提示替换非支持文档或改变上下文顺序仍可能影响 Reader。后续若开发风险控制，应针对 query-level intervention，而不是继续提升平均 support recall。

### 6.4 与 centralized reference 的差距恢复

Logistic ProbeRoute 对 centralized reference gap 的 Joint F1 恢复比例：

| 数据集 | FLAN | UnifiedQA |
|---|---:|---:|
| 2Wiki | 63.1% | 71.7% |
| HotpotQA | 131.6% | 123.1% |
| MuSiQue | 63.8% | 52.4% |

HotpotQA 超过 100% 是因为 ProbeRoute 在该冻结合同下已经超过 centralized reference。不能据此声称超过“centralized upper bound”，因为 centralized reference 不是 exhaustive/oracle retrieval，只是同一历史检索合同的集中式参考。

---

## 7. 审计、复现与运行状态

### 7.1 No-leak

最终 `no_leak_audit.json` 状态为 `pass`：

- Reader 在 R4 前未运行；
- final test 未访问；
- R3 packet 和 query manifests 记录 SHA-256；
- Reader 输入仅来自 materialized question、title 和 text；
- `answer`、`supporting_facts`、`is_supporting`、`gold_support` 不作为 Reader 推理特征；
- centralized reference 明确标记为 reference，而非 upper bound。

### 7.2 早期 BLOCKED audit 的解释

目录中的 `reader_preflight_audit.json` 是 2026-08-12 16:21 的保护性中间快照。当时 contexts 尚未物化，流水线被 SIGSTOP，状态为 `STOP_BEFORE_FIRST_CELL`。它记录的是修复前 checkpoint，不是最终实验状态。随后完成了冻结上下文物化、Reader contract audit 和 no-leak audit，才恢复流水线并运行 Reader。最终权威状态以以下文件为准：

1. `protocol/reader_context_contract_audit.md`：`pass`；
2. `protocol/no_leak_audit.json`：`pass`；
3. `reports/r4_reader_go_no_go.md`：`probe_route_end_to_end_confirmed`。

保留旧 BLOCKED 文件是为了展示保护动作与审计轨迹，不应将其误读为当前 blocker。

### 7.3 完整性

- FLAN 正式结果：3,600 rows；
- UnifiedQA 正式结果：3,600 rows；
- all contexts：3,600 rows；
- main result：24 data rows；
- paired bootstrap：54 comparisons；
- 运行结束后无活动 R4 process；
- PyTorch 可识别 4 张 CUDA GPU，任务结束时显存分配为 0；
- `nvidia-smi` 的 NVML mismatch 仍是系统监控问题，但未阻止 PyTorch CUDA 完成推理。

### 7.4 已修复的分析问题

1. BH 循环错误 `ValueError: too many values to unpack`：已修复；
2. aggregate 表按 query 重复写 300 次：已修复，最终主表为 24 行；
3. `two_sided_p` 可能大于 1：已改为 bootstrap tail probability + plus-one correction；
4. 修复后重新生成统计表，主状态及所有均值/CI 保持不变。

---

## 8. 论文层面的意义

R4 为 V20 提供了此前 HP4/V17 系列长期缺失的一段证据链：

> query-conditioned compact probes 改变客户端选择，提升有限通信预算内的多跳证据完整性；这种 evidence rescue 在冻结上下文与冻结双 Reader 条件下稳定提高 Joint F1。

论文不应只写成“一个更好的 client ranker”。更有说服力的主线是：

1. **问题定义：** Federated multi-hop retrieval 的关键障碍是资源选择和跨客户端 evidence co-access，而不是单一全局 relevance ranking。
2. **方法：** ProbeRoute 使用轻量、query-conditioned、可压缩的客户端 probe signals，在固定 `Bc=3` 和 15-document transmission budget 内进行资源选择。
3. **系统性质：** 不需要读取客户端全文，不需要第二轮 reader feedback，不修改 local retriever，不扩大客户端接触预算。
4. **实证：** 三数据集 retrieval 提升；双 Reader 端到端 Joint F1 6/6 cell 显著为正。
5. **机制：** support rescue queries 上 Reader Joint F1 大幅提升，且 rescue 数量远大于 harm。
6. **边界：** learned logistic 相对 label-free 的增量有限；final test 尚未打开；centralized reference 不是上界。

这条叙事比“所有指标都打败 centralized”更准确，也更贴合 Federated Search / Federated RAG 的论文贡献。

---

## 9. 局限与风险

1. 每个数据集只有一个冻结 N=300 holdout；虽然 query-level bootstrap 已显示稳定信号，但 final test 尚未完成。
2. 两个 Reader 都属于 T5-large 系列，架构多样性仍有限；不能直接外推到 decoder-only LLM。
3. centralized reference 不是 exhaustive oracle，Hotpot 上 ProbeRoute 超过它不能被解释为超过集中式检索上界。
4. Logistic 相对 label-free 的额外 Joint 收益多数未显著，学习器的独立必要性仍需谨慎表述。
5. Complete-support 提升并不保证每个 query 的 Answer 提升；少量 support harm 和 context-order sensitivity 仍存在。
6. 当前统计 p-value 来自有限 5,000 次 bootstrap，最小非零报告值为 0.0004；不应写成精确的极小概率。
7. 本轮没有重新测量完整端到端网络时延；通信与 probe payload 应引用 R3 frozen cost audit，不能从 R4 Reader runtime 反推。

---

## 10. 决策与下一步

### 10.1 当前决策

**Decision：进入冻结与论文整合，不再根据 R4 holdout 调 ProbeRoute、Reader 或 context 组合。**

R4 已达到预注册 go 条件。继续在同一 300-query holdout 上搜索阈值、prompt 或 rerank 会降低 confirmatory 价值。

### 10.2 建议执行顺序

1. 冻结提交 `13091c6`、模型哈希、query manifest、context manifest 和 R4 统计表；生成统一 artifact checksum manifest。
2. 将 R3 retrieval 表、compact probe communication audit、R4 Reader 表整理为论文三张主表：Retrieval、End-to-End、Cost/Mechanism。
3. 明确论文 primary method 的命名：建议以 `ProbeRoute` 为方法族，label-free 与 logistic 为两个 operating variants。
4. 在不查看标签的前提下预注册 R5 final-test 单次执行；是否开启由导师决定。
5. 若 9 月投稿时间紧，优先完成论文和复现实验包，不再开新模型分支。
6. 若必须补一个实验，优先选择预算敏感性或通信-Pareto 的 frozen replay，而不是再训练更复杂 ranker。

---

## 11. 本地产物索引

本报告：

- `V7-HP-PAPER/v20_arc_fedsearch/reports/v20_r4_frozen_dual_reader_complete_report_20260812.md`

已从服务器同步到本地：

- `stage_r4_frozen_reader/statistics/main_reader_results.csv`
- `stage_r4_frozen_reader/statistics/paired_bootstrap.csv`
- `stage_r4_frozen_reader/statistics/bh_secondary_tests.csv`
- `stage_r4_frozen_reader/statistics/per_query_results.csv`
- `stage_r4_frozen_reader/mechanism/support_transition_analysis.csv`
- `stage_r4_frozen_reader/mechanism/reader_gain_given_support_rescue.csv`
- `stage_r4_frozen_reader/mechanism/context_change_analysis.csv`
- `stage_r4_frozen_reader/gap_recovery/gap_recovery.csv`
- `stage_r4_frozen_reader/protocol/no_leak_audit.json`
- `stage_r4_frozen_reader/protocol/reader_context_contract_audit.md`
- `stage_r4_frozen_reader/protocol/reader_preregistration.md`
- `stage_r4_frozen_reader/reports/r4_reader_go_no_go.md`
- `stage_r4_frozen_reader/reports/r4_full_experimental_report.md`
- `stage_r4_frozen_reader/reports/r5_final_test_recommendation.md`

远端权威目录：

- `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/`

GitHub：

- 仓库：`MaverickChen030603/pprag_fl_SPU`
- 分支：`main`
- 当前分析修复提交：`13091c6`

---

## 12. 给导师的简要结论

V20 已从 retrieval-only 诊断推进到冻结双 Reader 的端到端确认。ProbeRoute 在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 上都提高了证据完整性，并在 FLAN-T5-Large、UnifiedQA-T5-Large 上获得一致、配对显著的 Joint F1 提升。最强证据不是单个最高分，而是：六个 primary cells 全部为正、CI 全部不跨零，且 support-rescue 子集表现出大幅 Reader 收益。当前研究主线已经具备论文所需的“方法—检索—机制—端到端”证据链。下一阶段应冻结结果、完成论文表格与 final-test 预注册，而不是继续在当前 holdout 上调参。
