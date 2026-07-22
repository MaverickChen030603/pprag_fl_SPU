# V7-HP-PAPER V16 最终实验报告

**任务名称：** V7-HP-PAPER-v16-synergy-aware-action-composition  
**内部工程代号：** CompoRepair  
**报告日期：** 2026-07-22  
**实验阶段：** Phase-A Oracle Composition Landscape  
**机器判定：** `hold_or_redirect`  
**最终状态：** `analysis_paper_only`  
**最终冻结测试：** 未启封、未运行  
**Learned composer：** 未训练，符合预注册停止规则

## 1. 执行摘要

V16 检验一个比 V14 更窄、也更可证伪的问题：在冻结候选池、固定五文档预算和相同 reader 下，状态依赖的多步 atomic edits 是否能得到任何合法单步编辑都无法达到的 reader-effective context。

Phase-A 共完成 314,476 个 Exact100 reader-context evaluations，覆盖 HotpotQA、2WikiMultiHopQA、MuSiQue，以及 FLAN-T5-Large、UnifiedQA-T5-Large 两个 reader。六个 dataset-reader 单元的 mean StrictSynJoint 均为正，paired-bootstrap 95% CI 均排除 0。这证明 Oracle 层面的 strict composition synergy 真实存在，并可跨数据集和 reader 复现。

但预注册继续条件要求至少两个数据集在两个 reader 上均达到 composition-only positive rate >=10%。最终只有 MuSiQue 达标，2Wiki 为 5%-6%，HotpotQA 为 2%-5%。因此通用 composer 的机会密度不足，Checkpoint 1 未通过。项目未训练 composer，未运行 Top-20 rescue，也未触碰 final test。

结论不是“composition 不存在”，而是：**composition 能抬高 Oracle 上界，但在当前真实 Top-10 pool 下，只有 MuSiQue 有足够多只能由 composition 修复的 query。该现象适合分析论文，不足以支持通用 learned composition 方法论文。**

## 2. 科学问题与研究假设

V16 的主问题为：

> Can a composition of multiple bounded context edits expose reader-effective contexts that cannot be reached by independent ranking or any single edit?

该问题拆成两个必须同时成立的命题：

1. **Oracle feasibility：** best composed context 在 query 内严格超过全部合法 single edits。
2. **Opportunity density：** 足够比例的 query 只有 composition 为正，值得训练专用 composer。

V16 没有把显著性本身当目标。预注册要求同时观察 effect size、composition-only rate、跨数据集复现和双 reader 一致性。

## 3. 文献边界与创新审计

Primary-source 审计覆盖 SetR、Contextual Passage Utility、Generative Context Pair Selection、Influence Guided Context Selection、R-CPS、Context-Picker、RankRAG、RECOMP、MDR、Beam Retrieval、GenDec、dynamic-k retrieval、set prediction 与 offline RL。

审计后的可辩护差异仅限于：冻结 pool 与固定 K 下，对完整有序 context 执行状态依赖 atomic edit trajectory，并显式计算相对所有合法 single edits 的 StrictSyn。它不等同于一般 subset selection、pair scoring、multi-hop retrieval 或 RL passage selection。Context-Picker 是最接近的潜在碰撞，因此正式方法名与强创新 claim 在 Phase-A 中保持冻结。

详细材料：

- `literature/literature_search_protocol.md`
- `literature/closest_method_matrix.md`
- `literature/method_novelty_hypotheses.md`

## 4. 新冻结协议与泄漏控制

### 4.1 历史暴露审计

系统扫描 191 个 V1-V15 历史来源，保守识别 16,405 个 HotpotQA IDs、34,790 个 2Wiki IDs 和相应规范化问题。所有历史暴露 query 均从新 confirmatory protocol 排除。三个 parse warning 来自“未生成结果”的说明文本，不对应实验样本。

### 4.2 新拆分

每个数据集冻结 5,000 train、1,000 development、1,000 calibration、2,000 final_test。历史 ID overlap 和 normalized-question overlap 均为 0。MuSiQue 拆分前去除 42 个重复问题；2/3/4-hop 保留在各 split。

### 4.3 Final sealing

Final inputs 与 labels 物理分离。Phase-A 仅在 development 上离线调用 reader。Final labels 未启封，未用于架构、阈值、子组或报告选择。

### 4.4 静态 no-leak audit

推理可见代码禁止 gold answer、support labels、answer presence、sealed labels 等字段。静态审计状态为 `pass`。Artifact hashes 与 split membership audit 均通过。

## 5. Action Space 与 Strict Synergy

完整状态为 `C=[d1,d2,d3,d4,d5]`。实现 KEEP、REPLACE、SWAP、MOVE、DROP_ADD、STOP，最长 trajectory 为三步，每一步保持五文档预算且无重复文档。

对指标 `M`：

`Delta_M(tau) = M(C_tau) - M(C_0)`

`StrictSyn_M(tau) = Delta_M(tau) - max_{a in A1} Delta_M(a)`

其中 `A1` 为该 query 在相同 Top-10 pool 下的全部合法 single edits。只有最短深度 2-3 的 context 进入 composed 集合，防止单步可达 context 被重复编码为 composition。

主指标为 Answer、SP/Evidence 和 Joint。MuSiQue 没有与 HotpotQA 完全同构的 official Joint，故其 Joint 字段在论文中必须称为 constructed Answer-Evidence composite。

## 6. Candidate Pool 与 Oracle Search

### 6.1 真实检索池

不使用 gold injection、support injection 或随机 padding。在每个数据集 development 前 100 query 上：

| Dataset | Support recall@10 | Complete support@10 | Answer access@10 | Support recall@20 | Complete support@20 | Answer access@20 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.7950 | 0.6100 | 0.7500 | 0.8300 | 0.6700 | 0.8100 |
| 2Wiki | 0.6550 | 0.3100 | 0.5400 | 0.6625 | 0.3200 | 0.5400 |
| MuSiQue | 0.6308 | 0.2700 | 0.4800 | 0.6675 | 0.3200 | 0.5400 |

2Wiki 与 MuSiQue 的 complete-support 上限较低。Composition 只能重组 pool 内文档，不能修复 pool absence。

### 6.2 Top-10 exact search

每个 query 枚举 `C(10,5)=252` 个五文档 subset。每个 subset 最多生成 baseline-relative、CrossEncoder 与 bridge-first 三种固定顺序；同时枚举全部 legal single REPLACE/SWAP/MOVE，并为每个有序 context 求精确最短编辑深度。

首轮 256-context capped smoke 因错误 depth 标注与 composed truncation 作废，只保留为工程负例。修复后先跑 Exact20，再按预注册最少 100 query/cell 完成 Exact100。

## 7. Exact100 主结果

### 7.1 Joint / constructed composite

| Dataset | Reader | N | Single+ | Composed+ | Composition-only | Best-single delta | Best-composed delta | StrictSyn | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | FLAN | 100 | .350 | .410 | .060 | +.0893 | +.1239 | +.0346 | [.0176, .0552] |
| 2Wiki | UnifiedQA | 100 | .450 | .490 | .050 | +.1476 | +.1827 | +.0351 | [.0168, .0563] |
| HotpotQA | FLAN | 100 | .440 | .460 | .020 | +.1155 | +.1491 | +.0337 | [.0170, .0545] |
| HotpotQA | UnifiedQA | 100 | .410 | .460 | .050 | +.1129 | +.1434 | +.0305 | [.0109, .0523] |
| MuSiQue | FLAN | 100 | .320 | .420 | .110 | +.1240 | +.1896 | +.0656 | [.0368, .0972] |
| MuSiQue | UnifiedQA | 100 | .290 | .370 | .100 | +.1074 | +.1486 | +.0412 | [.0169, .0677] |

六个 StrictSyn CIs 全部高于 0，是本阶段最稳健的正结果。Best-composed 比 best-single 平均提高约 3.05 至 6.56 分。与此同时，六个 median StrictSyn 均为 0，说明收益由 15%-20% 左右的 positive-synergy queries 驱动。

### 7.2 Answer 与 SP/Evidence 分解

| Dataset | Reader | Answer StrictSyn | Answer 95% CI | SP/Evidence StrictSyn | SP/Evidence 95% CI |
|---|---|---:|---:|---:|---:|
| 2Wiki | FLAN | +.0272 | [-.0111, .0700] | +.0369 | [.0211, .0558] |
| 2Wiki | UnifiedQA | +.0133 | [-.0157, .0393] | +.0369 | [.0211, .0558] |
| HotpotQA | FLAN | +.0363 | [.0080, .0747] | +.0368 | [.0190, .0577] |
| HotpotQA | UnifiedQA | +.0510 | [.0144, .0953] | +.0368 | [.0190, .0577] |
| MuSiQue | FLAN | +.0621 | [.0173, .1123] | +.0412 | [.0180, .0672] |
| MuSiQue | UnifiedQA | +.0474 | [.0043, .0931] | +.0412 | [.0180, .0672] |

SP/Evidence 协同在六个单元稳定；Answer 协同在 HotpotQA 与 MuSiQue 显著，但在 2Wiki 不显著。因此不能将所有 Joint 改善解释为 answer-generation 改善。

### 7.3 子组观察

仅解读 N>=20 的分层。MuSiQue 的 77 个标注 2-hop query 在 FLAN/UnifiedQA 上 StrictSyn 分别为 +.0682 与 +.0325，CI 均高于 0。HotpotQA 83 个 bridge query 为 +.0285 与 +.0248。2Wiki bridge-comparison 和 comparison 多数为正，compositional 子组的 UnifiedQA CI 跨 0。

更高 hop 的 MuSiQue 样本在 Exact100 中不足，不能据此建立单调 hop claim。下一篇分析论文若研究 hop，应扩大预注册样本，而不是复用当前小子组挑选结论。

## 8. Go/No-Go 结论

Checkpoint 1 要求至少两个数据集在两个 reader 上同时满足：N>=100、composition-only >=10%、mean StrictSynJoint >0、CI lower bound >0。

| Dataset | StrictSyn replicated | Composition-only threshold | Pass |
|---|---|---|---|
| 2Wiki | Yes | No | No |
| HotpotQA | Yes | No | No |
| MuSiQue | Yes | Yes | Yes |

只有 MuSiQue 通过，因此机器判定 `hold_or_redirect`。依照预注册：

- 不训练 greedy/imitation/synergy-aware/offline-RL composer；
- 不评估 learned composer realization；
- 不运行 Top-20 rescue；
- 不启封 final test；
- 不生成方法论文主表或宣称通用 composition breakthrough。

## 9. 失败机制与研究价值

主要失败不是“composed contexts 没有更高 reader score”，而是“只有 composition 能修复的 query 不够多”。HotpotQA 与 2Wiki 的许多改善已经可由某个 single edit 获得；训练多步 policy 的额外方法复杂度难以由 2%-6% 的独占机会率支撑。

MuSiQue 的较高 composition-only rate 与更丰富候选结构一致，但当前样本不足以区分 hop 数、pool diversity、evidence dispersion 或 reader sensitivity 哪一项是原因。因此最有价值的后续问题从“如何训练 composer”变为“什么候选池与问题结构产生 composition-only opportunity”。

## 10. 有效结论、无效结论与论文定位

### 10.1 当前证据支持

1. 固定 budget 的多步 context edit 在 Oracle 层面可严格超过所有 single edits。
2. StrictSynJoint 跨三个数据集、两个 reader 方向一致且显著。
3. Composition-only opportunity 高度 dataset-dependent。
4. MuSiQue 是继续研究 composition mechanism 的合理数据集。

### 10.2 当前证据不支持

1. Learned composer 可实现 Oracle synergy。
2. Action composition 普遍优于 best single-edit policy。
3. 方法优于 old Full、CrossEncoder、RECOMP 或近邻 selector。
4. 组合收益只来自更高 hop 数。
5. Top-20 能提高 composition-only rate。
6. 推理成本、风险或最终 QA Pareto 更优。

### 10.3 投稿建议

不建议以通用新方法投 ARR/ACL 主线。可将该项目重构为 ECIR 风格分析论文：**When Does Action Composition Help Context Construction?** 主贡献应是严格协同定义、matched Oracle protocol、机会密度分解和 dataset-dependent negative result，而不是未训练的 composer。

## 11. 复现与产物

核心产物包括：

- `protocol/used_query_inventory.json`
- `protocol/dataset_split_manifest.json`
- `protocol/artifact_hashes.json`
- `protocol/no_leak_audit.json`
- `oracle_search/phase_a_exact100/results/oracle_action_results.csv`
- `oracle_search/phase_a_exact100/results/oracle_synergy_statistics.csv`
- `oracle_search/phase_a_exact100/results/oracle_synergy_subgroups.csv`
- `oracle_search/phase_a_exact100/results/oracle_single_vs_composed.md`
- `oracle_search/phase_a_exact100/results/composition_go_no_go_1.md`
- `oracle_search/phase_a_exact100/results/synergy_distribution.pdf`

所有关键决策均由机器可读结果生成。Final labels 保持密封；停止点严格遵守预注册。
