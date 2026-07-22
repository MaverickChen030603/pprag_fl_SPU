# V7-HP-PAPER V16 当前实验报告

**任务：** V7-HP-PAPER-v16-synergy-aware-action-composition  
**内部代号：** CompoRepair，仅用于工程管理，尚未冻结为论文方法名  
**报告日期：** 2026-07-22  
**当前阶段：** Oracle Composition Landscape 已完成，Checkpoint 1 已判定  
**最终状态：** `analysis_paper_only`  
**最终冻结测试：** 未启封、未运行

## 1. 研究目的

V16 研究的核心问题是：在固定五文档预算下，状态依赖的多步上下文编辑，是否能够构造任何单步编辑都无法达到、但对 reader 的 Answer、Evidence 和 Joint 结果更有效的上下文。

该问题直接回应 V14/V15 的三个证据缺口：旧动作空间中大量 query 不存在正修复动作；pair complementarity 与 two-document action 缺少独立端到端归因；旧 proxy utility 与真实 reader 结果存在目标错位。V16 不把更小的 p 值作为目标，而要求更大的 paired effect、跨数据集复现、双 reader 一致性和 matched ablation。

## 2. 文献边界与暂定创新假设

已完成 primary-source collision search。最接近的方法包括 [SetR](https://aclanthology.org/2025.acl-long.861/)、[Contextual Passage Utility](https://aclanthology.org/2025.ijcnlp-short.37/)、[Generative Context Pair Selection](https://aclanthology.org/2021.emnlp-main.561/)、[Influence Guided Context Selection](https://papers.neurips.cc/paper_files/paper/2025/hash/2a07497c94cd24b20faa3fd14d847037-Abstract-Conference.html)、[Context-Picker](https://arxiv.org/abs/2512.14465)、[RankRAG](https://arxiv.org/abs/2407.02485)、[RECOMP](https://proceedings.iclr.cc/paper_files/paper/2024/file/bda88ed2892f5e61c9a9bf215c566913-Paper-Conference.pdf)、[MDR](https://openreview.net/forum?id=EMHoBG0avc1) 与 [Beam Retrieval](https://aclanthology.org/2024.naacl-long.96/)。

当前可辩护的新颖性边界不是一般意义上的 subset selection 或 RL selection，而是：

1. 在冻结候选池和固定 K 下，对完整有序 context 执行状态依赖 atomic edit trajectory；
2. 显式比较 best composed context 与所有合法 single edits；
3. 将 StrictSyn 定义为 composed gain 减去 query 内 best-single gain；
4. 只有在 Oracle 与 learned composer 两阶段都成立后，才把 action composition 写成核心贡献。

Context-Picker 是目前最接近的碰撞项，因此正式方法名和强创新 claim 继续冻结。详细矩阵位于 `literature/closest_method_matrix.md`。

## 3. 新冻结协议

### 3.1 历史暴露审计

V1-V15 共扫描 191 个历史来源，未跳过超大文件。保守暴露清单包含：

| Dataset | 历史 query IDs | 历史规范化问题 |
|---|---:|---:|
| HotpotQA | 16,405 | 16,405 |
| 2WikiMultiHopQA | 34,790 | 34,789 |

三个轻量 parse warning 均来自一份仅记录“未生成结果”的三行说明文件，不对应实验 query。反复分析过的 7,405 条 HotpotQA validation 全部排除在 V16 confirmatory test 之外。

### 3.2 数据拆分

HotpotQA、2WikiMultiHopQA、MuSiQue 各自冻结：

| Split | 每数据集数量 | 用途 |
|---|---:|---|
| train | 5,000 | 模型与离线 reader supervision |
| development | 1,000 | 架构与 Oracle 分析 |
| calibration | 1,000 | gate/threshold 校准 |
| final_test | 2,000 | 最终一次性确认 |

所有数据集均通过 query-ID、规范化问题和文件哈希审计。历史 ID overlap 与历史 question overlap 均为 0。MuSiQue 在拆分前去除 42 个重复规范化问题，2/3/4-hop 均保留在各 split 中。final labels 与 input 物理分离，权限限制为只读，当前未启封。

### 3.3 No-leak contract

推理可见代码静态扫描禁止 `sealed_label`、`final_test_labels`、`is_supporting`、`gold_answer`、`answer_presence`、`supporting_facts` 与 `supporting_titles`。当前扫描状态为 `pass`，无命中。Oracle reader labeling 可以离线读取 development labels，但不会成为 composer 推理特征。

## 4. 方法与实现

### 4.1 Context state 与 atomic actions

状态为有序五文档上下文 `C=[d1,d2,d3,d4,d5]`，候选池冻结，文档不可重复。已实现：

- `KEEP`
- `REPLACE(i,d)`
- `SWAP(i,j)`
- `MOVE(i,j)`
- `DROP_ADD(i,d)`
- `STOP`

每条 trajectory 最长三步，每步后预算仍为五文档。Phase-A 的最短可达性 witness 使用 REPLACE/SWAP/MOVE；DROP_ADD 保留给后续 matched ablation，因为其 membership 变化已可由 REPLACE 表达。

### 4.2 Strict composition synergy

对指标 M：

`StrictSyn_M(tau) = Delta_M(tau) - max_{a in A1} Delta_M(a)`

其中 `A1` 是该 query 的全部合法单步编辑。报告 epsilon 为 0、0.01、0.02，并计算 composition-only positive rate、mean/median StrictSyn、paired bootstrap CI、hop/type 子组。

### 4.3 Top-10 Oracle search

对每个 query 的 Top-10 pool：

1. 枚举 `C(10,5)=252` 个五文档 subset；
2. 每个 subset 仅生成 baseline-relative、CrossEncoder、bridge-first 三种固定顺序；
3. 枚举全部合法单步 REPLACE/SWAP/MOVE；
4. 对每个候选求精确最短 trajectory，只有最短深度 2-3 的 context 进入 composed 分母；
5. 同一有序 context 若有多个表示，只保留最低深度，防止单步 context 冒充 composition synergy。

Top-20 将使用受控 beam width 8/16/32，属于 exhaustive Top-10 之后的下一阶段。

### 4.4 Reader 与 evidence evaluator

冻结两个 reader：`google/flan-t5-large` 与 `allenai/unifiedqa-v2-t5-large-1363200`。HotpotQA 与 2Wiki 报告 Hotpot-style Answer/SP/Joint；MuSiQue 报告 Answer 与段落 evidence，并将乘积型结果明确标记为 constructed Answer-Evidence composite，不称为官方 Joint。

冻结 support predictor 的 development F1 分别为：HotpotQA 0.3905、2Wiki 0.5024、MuSiQue 0.3862。它们仅使用 query-unit lexical/entity/rank 特征，不读取 gold support。

## 5. 检索池质量

在新的 development 前 100 query 上，严格 real-retrieval smoke 得到：

| Dataset | Support recall@10 | Support recall@20 | Complete support@10 | Complete support@20 | Answer access@10 | Answer access@20 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.7950 | 0.8300 | 0.6100 | 0.6700 | 0.7500 | 0.8100 |
| 2Wiki | 0.6550 | 0.6625 | 0.3100 | 0.3200 | 0.5400 | 0.5400 |
| MuSiQue | 0.6308 | 0.6675 | 0.2700 | 0.3200 | 0.4800 | 0.5400 |

因此 2Wiki/MuSiQue 存在明显 pool-absence 上限。V16 的 composition 只能重组已进入 pool 的文档，不能将检索缺失错误归因给 composer。

## 6. 已完成的 Oracle 结果

### 6.1 失效的 capped smoke

第一轮每 query 最多 256 contexts 的结果不能用于 Go/No-Go。审计发现 Top-10 exhaustive subset 被统一标为 depth 5，导致实际可由 2-3 步到达的 context 被排除；同时 256 cap 截断了 composed 候选。该结果仅保留为搜索实现的负向 smoke，不进入正式论证。

### 6.2 修正后的 20-query exact smoke

已对 62,396 个 reader-context pairs 完成双 reader 推理。每个 query 包含 baseline、全部单步编辑和所有固定顺序 subset，且不设 context cap。

| Dataset | Reader | N | Single positive | Composed positive | Composition-only | Mean StrictSyn Joint | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | FLAN | 20 | 0.400 | 0.450 | 0.050 | +0.0364 | [0.0000, 0.0964] |
| 2Wiki | UnifiedQA | 20 | 0.550 | 0.600 | 0.050 | +0.0614 | [0.0014, 0.1364] |
| HotpotQA | FLAN | 20 | 0.400 | 0.400 | 0.000 | +0.0124 | [0.0017, 0.0250] |
| HotpotQA | UnifiedQA | 20 | 0.350 | 0.350 | 0.000 | +0.0167 | [0.0000, 0.0476] |
| MuSiQue | FLAN | 20 | 0.050 | 0.200 | 0.150 | +0.0383 | [0.0000, 0.1000] |
| MuSiQue | UnifiedQA | 20 | 0.150 | 0.250 | 0.100 | +0.0375 | [0.0000, 0.1000] |

### 6.3 当前解读

修正后六个 dataset-reader 单元的 mean StrictSyn 均为正，说明精确深度恢复解决了 capped smoke 的主要假阴性。MuSiQue 在双 reader 上达到 10%-15% composition-only，符合“更大 hop 数和更丰富候选池更可能需要组合”的机制预期。2Wiki 只有 5%，Hotpot 为 0%，尚不支持跨数据集主张。

20-query CI 离散且样本太小，不能执行 Checkpoint 1。机器状态为 `insufficient_sample`，不是 `continue_composition`。在读取该结果前已冻结每个 dataset-reader 至少 100 query 的门槛。

### 6.4 100-query exact Top-10 正式 Oracle 统计

Exact100 共完成 314,476 个 reader-context evaluations。每个 dataset-reader 单元均包含 100 个 development query，统一使用相同 Top-10 pool、五文档预算、全部合法 single edit 和固定三种 subset ordering。

| Dataset | Reader | Single+ | Composed+ | Composition-only | Best-single | Best-composed | Mean StrictSyn Joint | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | FLAN | 0.350 | 0.410 | 0.060 | +0.0893 | +0.1239 | +0.0346 | [0.0176, 0.0552] |
| 2Wiki | UnifiedQA | 0.450 | 0.490 | 0.050 | +0.1476 | +0.1827 | +0.0351 | [0.0168, 0.0563] |
| HotpotQA | FLAN | 0.440 | 0.460 | 0.020 | +0.1155 | +0.1491 | +0.0337 | [0.0170, 0.0545] |
| HotpotQA | UnifiedQA | 0.410 | 0.460 | 0.050 | +0.1129 | +0.1434 | +0.0305 | [0.0109, 0.0523] |
| MuSiQue | FLAN | 0.320 | 0.420 | 0.110 | +0.1240 | +0.1896 | +0.0656 | [0.0368, 0.0972] |
| MuSiQue | UnifiedQA | 0.290 | 0.370 | 0.100 | +0.1074 | +0.1486 | +0.0412 | [0.0169, 0.0677] |

六个单元的 mean StrictSyn Joint 均为正，paired bootstrap 95% CI 均排除 0，因此“多步 context 在 Oracle 层面可以严格超过全部合法单步编辑”获得跨数据集、跨 reader 复现。不过其中位数均为 0，说明增益集中在少数 query，而非普遍发生。

composition-only positive rate 的结论不同：2Wiki 为 5%-6%，HotpotQA 为 2%-5%，只有 MuSiQue 在两个 reader 上分别达到 11% 与 10%。因此严格协同效应真实存在，但 action composition 不是三个数据集上的主要 opportunity source。

### 6.5 Answer、Evidence 与子组边界

SP StrictSyn 在六个单元均显著为正。Answer StrictSyn 在 HotpotQA 与 MuSiQue 的四个单元显著为正，但 2Wiki 两个 reader 的 CI 均跨 0。这表明 2Wiki 的 Joint 协同主要来自 evidence 改善，不能扩写为稳定 answer gain。

样本量至少 20 的子组中，MuSiQue 2-hop 在 FLAN 与 UnifiedQA 上的 Joint StrictSyn 分别为 +0.0682 和 +0.0325；HotpotQA bridge 主组分别为 +0.0285 和 +0.0248。2Wiki bridge-comparison 与 comparison 多数为正，但 compositional 子组的 UnifiedQA CI 跨 0。由于 Exact100 的更高 hop 子组样本不足，本阶段不声称“hop 越高 composition 越有效”。

## 7. Checkpoint 1 判定

预注册规则要求至少两个数据集同时满足：每个 reader 至少 100 query、composition-only >=10%、mean StrictSynJoint >0、paired CI lower bound >0。数据集级判定为：

| Dataset | 双 reader StrictSyn 显著 | 双 reader Composition-only >=10% | Dataset pass |
|---|---|---|---|
| 2Wiki | 是 | 否 | 否 |
| HotpotQA | 是 | 否 | 否 |
| MuSiQue | 是 | 是 | 是 |

机器输出为 `hold_or_redirect`。只有一个数据集通过，未达到“至少两个数据集”要求。因此：

1. 不进入 composer 训练；
2. 不运行 learned composer、greedy composer 或 offline RL；
3. 不启封 final frozen test；
4. 不以 Top-20 或更多搜索继续追逐预注册门槛；
5. V16 最终科学状态选择 `analysis_paper_only`。

## 8. 科学解读与后续路线

V16 同时建立了一个正结论和一个限制结论。正结论是：在固定 pool、固定预算和严格 best-single 对照下，Oracle composition synergy 跨三个数据集和两个 reader 稳定存在。限制结论是：真正只能由 composition 修复的 query 比例在 HotpotQA 与 2Wiki 很低，尚不足以支撑通用 learned composer 的方法论文。

更合适的论文问题是：**When Does Action Composition Help Context Construction?** 该分析主线可研究 pool absence、single-action opportunity、composition-only opportunity 与 hop/pool structure 的关系。若未来重新启动方法开发，应先扩大真实 candidate opportunity 或训练 direct set utility，再在新的独立 protocol 上重做 Checkpoint 1，而不是在当前 development 结果上继续调 composer。

## 9. 当前结论

V16 已完成文献边界、历史暴露审计、三数据集新冻结协议、动作系统、严格协同定义、真实候选池、双 reader 与 Exact100 Oracle 分析。最终最准确的表述是：**Oracle action composition 能显著改善 best-single 上界，但 composition-only opportunity 只在 MuSiQue 达到预注册门槛；通用 composer 主线未通过 Checkpoint 1，故停止训练并转为分析论文路线。**
