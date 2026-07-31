# V17/V18 Federated Action RAG Checkpoint-A 完整实验报告

**面向对象：** 导师讨论与后续研究决策
**报告日期：** 2026.07.31
**实验名称：** V7-HP-PAPER-v17-federated-action-rag / FedAction-RAG Phase-A
**后续协议层：** V18 Opportunity-Gated FedAction
**正式运行标识：** `phase_a_checkpoint100`，冻结起点 `fac9f62`
**结论状态：** `hold_or_redirect`；V18 映射为 `checkpoint_a_fail`

---

## 摘要

本实验不是直接训练一个新的联邦 RAG 模型，而是先回答一个更基础的
可行性问题：在严格的客户端访问预算下，自然形成的知识 silo 是否会
产生足够多的、只能通过跨客户端组合上下文才能改善下游多跳问答 reader
的机会？若这一前提不成立，后续引入 composer、FedAvg/FedProx/SCAFFOLD、
个性化 adapter 或 selective upload 都可能只是在优化一个缺少真实 reader
收益的 Oracle 搜索空间。

因此，V17 在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 上，以两种冻结
reader、固定的 Bc、local-k、K=5 和十文档动作池，对中心化、topic-silo、
entity-community 与随机分片进行了 30 个完全匹配的 Oracle 单元评估。所有
单元均为 N=100，使用相同 query、reader、候选预算和评测代码。V18 随后对
运行进行独立完整性审计，并统一整理指标与分支决策。

实验得到一个清晰但并非消极无价值的结论：**自然 topic silo 的 source
routing 的确提升了 Bc=3 下可获得的支持证据，且相对随机分片在若干
reader 上提高了 composition-only 机会率；然而，这些机会未能稳定转化为
跨 reader、优于中心化对照的 CrossClientStrictSyn。** 三个数据集没有任何
一个通过预注册的联合门槛。因此，协议禁止进入通用 learned composer 的
Phase B、联邦个性化的 Phase C，以及面向 FedAction 的参数 selective-upload
Phase D。最合理的后续方向是回到已有正信号基础更扎实的
reader-aligned selective retriever-block upload 主线，同时保留本轮构建的
source-routing、通信与双 reader 评价资产。

---

## 1. 研究背景与问题

### 1.1 动机

此前 V7-HP 系列已经表明：仅在上游检索 proxy 或 block 选择层观察到差异，
并不保证差异能穿过联邦聚合、dense representation、top-K truncation 与
reader 生成，最终体现为正式 QA 收益。HP4 的 controlled micro-benchmark
曾验证软路由、混合检索与 reader-aware counterfactual credit 的机制潜力，
但在严格 no-leak 的真实验证集上，收益明显缩小。

V17 因而将问题前移：在训练任何更强策略之前，先验证联邦数据局部性是否
真的带来 reader 可利用的跨客户端上下文互补性。这样避免把“分布式访问”
与“分布式 context composition 的下游收益”混为一谈。

### 1.2 Phase-A 研究问题

**RQ-A：** 在固定客户端预算 Bc<=3、固定 local-k=5、固定 K=5 reader
context 和相同十文档动作池条件下，natural topic client partitions 是否
比中心化与预算匹配随机分片创造更多 *Bc-realizable cross-client
composition-only* 的 reader 增益？

预注册假设为：

- **H1：** natural partitions 比中心化和随机分片具有更高的可实现跨客户端
  composition opportunity。
- **H2：** 该机会集中于 gold evidence 实际分散至多个 client 的 query。
- **H3：** 该机会在 FLAN-T5-Large 与 UnifiedQA-T5-Large 上方向一致，且
  Bc=2 有正向的次级证据。

这里的 Phase-A 是 Oracle feasibility checkpoint，不是部署模型评估。它回答
“后续值得学习什么”，而非声称已训练出可以部署的 router 或 composer。

---

## 2. 预注册决策规则

在读取正式 reader 输出前，V17 将后续模型训练锁定为一个联合 Go/No-Go：

1. 至少两个数据集、两个 reader 的 natural topic-silo Bc=3 单元都需满足
   CrossClientStrictSyn 的 95% bootstrap CI 下界大于 0；
2. 两个数据集的 cross-client composition-only rate 均需 >=10%；
3. 该 rate 需要在同一 query 上同时显著高于 central 和 random-balanced
   control；
4. 收益应集中在真实跨 client evidence query；
5. Bc=2 至少给出方向一致的可行性证据，或有明确的 client-recall 原因；
6. 两 reader 方向一致。

任一前序 checkpoint 未通过，不允许通过调大 Bc、pool、local-k、改变 query
ID、选择更有利 partition 或启动更复杂训练来“救回”结论。该设计的目的，是
让负结论也具备可解释性与研究价值。

---

## 3. 数据、分割与 no-leak 设计

### 3.1 数据集与冻结 split

核心数据集为 HotpotQA、2WikiMultiHopQA 和 MuSiQue。每个数据集从 V1--V16
已使用 query ID 与归一化 question 中剔除重合项，形成新的 train/development/
calibration/final-test 划分；其中 development 包含 1,000 个候选 query，
Phase-A 按固定前 100 个冻结 query 运行。final-test 输入与标签保持封存，
不参与本轮设计、路由、动作生成和报告决策。

### 3.2 标签隔离

下列字段禁止进入可部署路由、候选生成与将来的 gate 特征：gold answer、
answer presence、supporting titles/facts、support client、reader-generated
target answer 和 final-test labels。gold evidence 仅用于 Oracle 后验的机会
归因与评测，不能用于推理时特征。静态 no-leak 审计覆盖 retrieval、routing、
centralized selector、federated training、risk gate、communication、baseline
与 evaluation 目录，最终结果为 pass。

### 3.3 客户端划分与 query origin

- **M=20 clients。** 每个文档仅属于一个 client。
- **topic-silo（主自然 partition）：** 基于文档主题 embedding/聚类构建。
- **entity-community（预注册 replication）：** 以实体共现 community 划分。
- **random-balanced：** 保持客户端数量及 Bc 访问数匹配的负对照。
- **centralized：** 相同 query、reader、K=5 和动作搜索代码下的无联邦对照。
- **query origin：** 由 query-client topic similarity 的确定性采样赋值，
  seed 为 20260723；origin 始终包含在可路由 client 集中。

---

## 4. 实验配置与执行管线

### 4.1 固定配置

| 类别 | 冻结配置 |
|---|---|
| 正式样本 | 每 dataset-reader-condition N=100，共 30 单元 |
| 条件 | centralized；topic Bc=2；topic Bc=3；entity Bc=3；random Bc=3 |
| client budget | Bc=3 主实验，Bc=2 次级检验；centralized Bc=1 |
| 检索深度 | 联邦条件 local-k=5；中心化 control local-k=10 |
| 动作候选池 | 10 文档；保留 client-local 候选以供审计 |
| 最终 context | K=5 文档 |
| dense/sparse | BAAI/bge-base-en-v1.5 + SQLite FTS5 BM25；0.55 dense + 0.45 sparse |
| readers | google/flan-t5-large；allenai/unifiedqa-v2-t5-large-1363200 |
| 统计 | paired bootstrap 5,000；95% CI；双侧 paired randomization p；控制比较 BH 校正 |

### 4.2 实际执行步骤

1. 为每种 partition/client 建立物理独立 SQLite FTS shard，避免先做 global
   retrieval 再过滤所造成的伪 local search。
2. 在 Bc 约束下 router 选择 source clients；各 client 返回 local top-k，
   形成可审计的 candidate pool。
3. 从十文档 action pool 构造 single-client context、single cross-client edit、
   多文档 cross-client composition、within-client composition 及 baseline
   trajectory；每个动作输出完整 K=5 context。
4. 两个 reader 对每条 trajectory 离线生成 answer，并以冻结 support predictor
   计算 Answer F1、SP F1、Joint F1（MuSiQue 使用 answer-evidence composite，
   不称为官方 Joint）。
5. 对同一 query 的 trajectory 进行 Oracle 汇总、条件比较与机器 Go/No-Go。

### 4.3 指标定义

对每个 query 和指标 M：

```text
BestSingleClient_M      = 最优单客户端 context 相对 baseline 的增量
BestSingleCross_M       = 最优单一跨客户端 edit 的增量
BestCrossComposition_M  = 最优多文档跨客户端 composition 的增量

CrossClientStrictSyn_M = BestCrossComposition_M
                        - max(BestSingleClient_M, BestSingleCross_M)
```

若 `BestCrossComposition_M > 0` 且两个 single comparator 均 `<=0`，则该 query
记为 **cross-client composition-only**。该定义排除了“单文档替换本来就能
解决”的情形，直接询问联合 composition 是否带来额外 reader 价值。

---

## 5. 完整性、可复现性与运行状态

V18 的 post-completion integrity audit 对本次输出执行了 104 项检查，全部
通过：

- 30 个 reader-backed 单元均存在，且每单元恰为 100 个唯一 query；
- 同一数据集的五个条件和两个 reader 的 query 集完全对齐；
- partition、Bc、local-k、pool-size、K=5、client ID 范围与 action context
  约束一致；
- 文档在每个 pool 内无重复；context 文档来自对应候选 pool；
- formal start provenance 为 `fac9f62`；所有 pool/context/per-query 文件已
  生成 SHA-256 hash；
- static no-leak audit 为 pass；无 crash-recovery 分支，因此没有任何
  重启后用不同配置替换原结果的情形。

实现过程中发现并修复了一个纯聚合器输入过滤问题：路由审计 CSV 与
reader per-query CSV 共置时，前者不含 `reader` 字段。修复提交为 `fac9f62`，
并在正式 N=100 运行开始前冻结；不会改变 reader 输出或实验配置。随后对
V18 审计器又修正了“完整 local candidate retention”与“十文档 action pool”
的表示区别：pool 记录可以保留多于十个 local candidates 用于审计，但
action generation 仍严格截取 top-10。该修正只影响事后审计逻辑，不改变
V17 正式数据。

---

## 6. 主要结果

### 6.1 主条件：topic-silo，Bc=3

| Dataset | Reader | StrictSyn | 95% CI | Paired p | Composition-only | Cross-client evidence | 完整 support 位于 action pool |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | FLAN | +0.0338 | [-0.0098, +0.0753] | 0.1200 | 7% | 43% | 41% |
| HotpotQA | UnifiedQA | +0.0292 | [-0.0078, +0.0648] | 0.1118 | 4% | 43% | 41% |
| 2Wiki | FLAN | -0.0342 | [-0.0895, +0.0197] | 0.2138 | 5% | 63% | 20% |
| 2Wiki | UnifiedQA | +0.0116 | [-0.0399, +0.0639] | 0.6387 | 9% | 63% | 20% |
| MuSiQue | FLAN | +0.0234 | [-0.0248, +0.0703] | 0.3501 | 10% | 74% | 12% |
| MuSiQue | UnifiedQA | +0.0288 | [-0.0133, +0.0716] | 0.2048 | 11% | 74% | 12% |

主结果可概括为：

- HotpotQA 两 reader 的点估计均为正，但 CI 均跨 0，composition-only rate
  仅 4--7%，低于 10% 门槛。
- 2Wiki 的 FLAN 为负，UnifiedQA 接近 0，两个 reader 不构成稳定正信号。
- MuSiQue 的 composition-only 达到 10% 和 11%，但 StrictSyn CI 仍跨 0；
  其可获得 evidence 很多，真正能在十文档池中组成完整支持链的比例却只有
  12%。

因此，没有任一 dataset-reader cell 满足 CI 条件，更没有数据集在两 reader
上通过。

### 6.2 Bc=2 次级检验

| Dataset | FLAN StrictSyn (95% CI) | Comp-only | UnifiedQA StrictSyn (95% CI) | Comp-only |
|---|---:|---:|---:|---:|
| HotpotQA | -0.0111 [-0.0635, +0.0373] | 6% | -0.0233 [-0.0706, +0.0206] | 2% |
| 2Wiki | -0.0519 [-0.1057, +0.0015] | 4% | -0.0078 [-0.0582, +0.0425] | 8% |
| MuSiQue | -0.0131 [-0.0593, +0.0312] | 7% | +0.0006 [-0.0402, +0.0397] | 7% |

Bc=2 没有提供方向一致的支持。这个结果排除了“只要把模型设计得更好，低
source budget 下就自然可以恢复同一机会”的乐观解释：在当前 routing 和
candidate contract 下，预算本身已经使可实现 composition opportunity 进一步
变稀薄。

### 6.3 与随机和中心化 control 的比较

topic-silo 相对 random-balanced 的 StrictSyn 差异在多个 reader 中为正，
例如 Hotpot-FLAN +0.2420（CI [+0.1818, +0.3076]）、2Wiki-UnifiedQA +0.1315
（CI [+0.0703, +0.1957]）和 MuSiQue-FLAN +0.1464（CI [+0.0872, +0.2091]）。
composition-only rate 的随机对照差异中，MuSiQue-FLAN 为 +10pp（CI
[+5pp, +16pp]），2Wiki-UnifiedQA 为 +8pp（CI [+3pp, +14pp]），Hotpot-FLAN
为 +6pp（CI [+1pp, +11pp]）。这些结果证明 natural topic structure 并非和
random sharding 完全等价。

但相对 **centralized** 的 composition-only 差异为：Hotpot 0pp/1pp、2Wiki
0pp/1pp、MuSiQue -1pp/0pp（FLAN/UnifiedQA）；所有区间均跨 0。严格说，
topic-silo 没有显示出超过中心化十文档池的 reader-level 组合机会。这正是
Checkpoint-A 未通过的决定性原因：随机对照优势只能说明分区结构影响 local
routing，不足以证明联邦限制产生了独特且可学习的 reader 价值。

### 6.4 路由与证据可得性诊断

下表给出 topic Bc=3 与 random Bc=3 的 action-pool 完整支持率（complete
support in action pool）：

| Dataset | Topic Bc=3 | Random Bc=3 | Centralized | Topic 的主要路由变化 |
|---|---:|---:|---:|---|
| HotpotQA | 41% | 1% | 66% | gold-client recall 86%，完整 evidence-client recall 76% |
| 2Wiki | 20% | 1% | 42% | gold-client recall 65%，完整 evidence-client recall 48% |
| MuSiQue | 12% | 1% | 33% | gold-client recall 62%，完整 evidence-client recall 39% |

这组诊断非常重要。随机划分会把 support 打散到大量无语义 client，使 Bc=3
下完整支持几乎无法进入 action pool；topic routing 明显改善了这一点。但
centralized 仍有更高的完整支持率，且 topic pool 中的 evidence 质量、排序和
答案表述锚点不一定足以让 reader 完成推理。也就是说，**source routing
success != context composition success != reader success**。

### 6.5 通信与效率观察

联邦 Bc=3 条件平均联系 3 个 client、传输约 15 个文档；Bc=2 约为 2 个
client、10 个文档。中心化约 10 个文档。retrieval latency 表现出系统开销：
Hotpot/2Wiki 的联邦物理 shard 路径约 24--34 秒/查询，而中心化约 10--12 秒；
MuSiQue 由于索引规模和实现路径不同，联邦约 6 秒/查询、中心化约 11 秒。
因此，即使未来存在小质量收益，也必须与 source contact、文档传输和延迟
共同报告，不能只以 Oracle quality 作结论。

---

## 7. 结果解释：为什么有路由信号但没有足够的 reader 信号？

### 7.1 可得性瓶颈没有完全消除

topic routing 在 Hotpot 的完整支持池覆盖达到 41%，但 2Wiki 和 MuSiQue 仅为
20% 和 12%。这意味着大部分 query 即使选择了“正确的”clients，也没有在
top-5 local retrieval、十文档截断和 K=5 context 中同时保留必要 hop。任何
训练 composer 的有效正例分母都比跨-client evidence rate 小得多。

### 7.2 Oracle action space 本身并不保证回答可读性

Oracle 在动作空间内比较 reader 输出，因此严格优于单 edit 的 trajectory 已
经是事后最优选择。即便在这个上限设定中，StrictSyn 仍不稳定，表明瓶颈不
只是“selector 尚未学好”。多跳 supporting facts 的共现可能提高 SP proxy，
却可能删去 reader 回答所需的 lexical answer anchor，或使证据顺序不利于
generation。这与 HP4 中“support 提升而 answer/joint 不一定同步”的诊断一致。

### 7.3 中心化对照否定了“联邦特有机会”的强主张

natural topic partitions 确实比 random partition 更有结构，但中心化的固定
十文档池并未被自然分区稳定超越。若继续直接训练 Federated Action composer，
模型很可能只是在学习如何补偿分区造成的缺失，而不是利用一个中心化方法
无法获得的优势；这会削弱方法新颖性与实际价值。

### 7.4 Bc=2 的失败提示低预算下的设计风险

主张 FedAction 的一个必要条件是 Bc<=3，而 Bc=2 的结果整体为负或近零。
这说明当前实验中机会对 client budget 极为敏感。没有 Bc=2 的方向性支持时，
把 Bc=3 的局部点估计推广为一般 low-communication federated method 是不
稳健的。

---

## 8. 文献阅读与定位

本周的文献阅读不以“所有 Federated RAG 都是 baseline”为前提，而按方法的
**决策对象**进行重分类。完整矩阵见
`literature/decision_object_matrix.md`。

### 8.1 Training-time model-update selection

FedAvg、Federated Dropout、Adaptive Federated Dropout、FedPAQ 与旧 V7-HP
selective upload 的选择对象是模型参数、子网络或 parameter block，发生在
训练通信阶段。它们回答“相同 payload 下传哪些更新”，而不是“当前 query
给 reader 传哪五个文档”。这条线与本项目原始 V7-HP 正向目标直接相关，
也是本报告推荐回归的主线。

### 8.2 Federated retriever adaptation 与 personalization

pFedRAG、FedRAG framework 和 FedE4RAG 主要学习 retriever embedding、
RAG component 或 shared/personalized layers。它们证明非 IID 下可以做联邦
retriever adaptation，但不证明 complete reader-context action 的跨 client
synergy。V17 未通过意味着尚不能以“个性化 action utility”作为下一步主张；
若转向个性化，先应在 selective upload/retriever 层验证。

### 8.3 Query-time federated search 与 result merging

RAGRoute / Efficient Federated Search、FeB4RAG 和 DRAG 的核心对象是 source/
peer route 或 source results。它们对本项目最直接的启发是：client recall、
budget-matched random control、documents transferred 和 latency 都应成为一等
指标。V17 的 routing 诊断正是这一类工作与 reader-aware QA 之间必要的桥梁。
但本实验显示 source selection 的改善不能自动等价于 downstream QA 改善。

### 8.4 Aggregation、memory 与 context selection

FedMosaic 在 adapter/parameter space 讨论 selective aggregation 与 conflict；
FD-RAG 使用 distributed memory 的 fast/slow path；HyFedRAG 与 Federated
In-Context Agent Learning 使用不同的 representation 或 agent-state contract。
这些工作可以启发未来冲突控制与系统设计，但不能被不公平地作为同一固定
K=5 document action 的数值对照。

在另一侧，SetR、Context-Picker、contextual passage utility 与 influence-guided
selection 已经证明 centralized reader-aware set selection 的价值。因而本项目
不能声称“first reader-aware selector”。一个可能的、但目前尚未成立的差异化
主张只能是：在 **预算化跨 client retrieval + complete context action +
federated local training** 三者同时成立时，学习 reader-aligned utility。
Phase-A 的失败恰恰阻止了在缺少这三个条件联合证据时过度宣称。

RAGAS 只适合作为 context relevance、faithfulness、answer relevance 的辅助
自动指标，不能替代 HotpotQA/2Wiki/MuSiQue 的 answer/evidence native metrics；
MSRS 则可在核心多跳 QA 方法成立后作为多源长文本检索-综合外部验证，而不应
用来回避当前的基本 reader-synergy 缺口。

---

## 9. 有效性、统计与局限

### 9.1 内部有效性优势

- 预注册的联合门槛避免事后选择某一有利 dataset、reader 或 partition；
- central 与 random control 均使用同 query、Bc/local-k/K/reader/action-search
  合同；
- 两个结构不同的 reader 减少了单 reader 偶然性；
- query-level paired bootstrap 与随机化检验比只报告 aggregate mean 更合适；
- 104/104 integrity checks 和 no-leak audit 使“失败”不能轻易归因于单元缺失
  或 label leakage。

### 9.2 需要谨慎的地方

1. **Oracle 的解释边界。** Oracle 使用离线 reader 结果选择最佳动作，反映
   动作空间上限，不等于可部署 selector；本结果的“无稳定 synergy”反而比
   学习器失败更早、更强地限制了后续方法。
2. **N=100 的精度。** 这是正式最小样本而非大规模 confirmatory sample；CI
   跨 0 的结论说明不能主张稳定增益，但不能证明任何极小效应绝对不存在。
3. **单一 retriever/动作生成器。** 冻结 BGE+BM25 与 V16 action generator
   有利于因果隔离，却不代表所有 retriever、pool size 或语言模型组合。
4. **natural partition 定义。** topic 和 entity partitions 是合理的模拟，
   但不等价于真实机构的数据治理、时间漂移或跨组织知识分布。
5. **效率实现差异。** 各数据集的 index 规模和物理 FTS shard 路径不同，
   latency 适合作为本系统的测量结果，不应被泛化为所有 federated search
   系统的固定成本。
6. **多重比较。** 控制对比已提供 BH 修正；主结论没有依赖单个未校正的
   positive cell，而依赖预注册的跨数据集、跨 reader 联合失败。

---

## 10. 最终 Go/No-Go 与建议

### 10.1 机器决策

| Checkpoint | 结论 | 依据 |
|---|---|---|
| Phase-A | `hold_or_redirect` | 0/3 datasets 通过双 reader 联合规则；没有 primary CI 下界 >0；中心化对照条件失败；Bc=2 不稳定 |
| V18 branch | `checkpoint_a_fail` | 104/104 integrity pass 后的预注册映射 |
| Phase-B centralized composer | 不启动 | Phase-A 不支持足够可学习的 reader-beneficial opportunity |
| Phase-C FL/personalization | 不启动 | 不允许以更复杂的训练掩盖前序机会失败 |
| Phase-D FedAction selective upload | 不启动 | 仅在 Phase-C 有效后才允许 |

### 10.2 推荐的研究转向

**推荐状态：`return_to_selective_upload`。**

1. 回到 V7-HP 的 reader-aligned selective retriever-block upload，严格对齐
   payload、rounds、client participation 和 initialization；将 V17 的 source
   routing、documents transferred、latency 与双 reader QA 指标加入其正式评价。
2. 先验证“相同训练通信 payload 下，reader-aligned update selection 是否保持
   retriever/routing/QA 表现”，而不是再直接构造一个没有通过机会门槛的
   FedAction composer。
3. 若导师认为高分散 query 仍值得研究，应建立一个**新的、独立冻结的
   Phase-A2 development protocol**：预先定义只用 inference-safe router
   entropy、client score margin、entity overlap、retrieval statistics 与
   question complexity 的 opportunity detector；不能用 gold evidence 来
   筛选部署子集，也不能重新解释本轮 primary failure。
4. 论文写作上，可保留本轮作为方法学/负结果资产：它说明 federated source
   access、retrieval opportunity 与 reader-level utility 需要分层审计。但它
   目前不足以支撑一篇以通用 FedAction-RAG 方法为中心的强方法论文。

---

## 11. 关键产物与复现入口

### V17

- `V7-HP-PAPER/v17_fedaction_rag/protocol/v17_preregistration.md`
- `V7-HP-PAPER/v17_fedaction_rag/oracle/phase_a_checkpoint100/results/federated_go_no_go_phase_a.json`
- `V7-HP-PAPER/v17_fedaction_rag/oracle/phase_a_checkpoint100/results/federated_oracle_results.csv`
- `V7-HP-PAPER/v17_fedaction_rag/oracle/phase_a_checkpoint100/results/partition_control_results.csv`
- `V7-HP-PAPER/v17_fedaction_rag/oracle/phase_a_checkpoint100/results/routing_metrics_summary.csv`

### V18

- `V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a/01_checkpoint_a_integrity_audit.py`
- `V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a/checkpoint_a_integrity.json`
- `V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a/checkpoint_a_all_units.csv`
- `V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a/checkpoint_a_partition_comparison.csv`
- `V7-HP-PAPER/v18_opportunity_gated_fedaction/literature/decision_object_matrix.md`
- `V7-HP-PAPER/v18_opportunity_gated_fedaction/reports/checkpoint_a_result_report.md`

### Git history

- `fac9f62` -- Phase-A aggregation input contract fixed before formal run;
- `2ae03dd` -- V18 preregistration and audit scaffold;
- `6778dbe` -- candidate-retention/action-pool audit clarification;
- `7926d54` -- concise checkpoint failure record;
- `4d7848d` -- weekly research report.

---

## 12. 一句话结论

**V17 严格证明了自然分区可改善预算化 source routing 的证据可得性，但未能
证明这种改善在当前 Bc<=3、K=5 和双 reader 合同下形成稳定、优于中心化的
跨客户端 reader synergy；因此应停止通用 FedAction composer，转回
reader-aligned selective upload。**
