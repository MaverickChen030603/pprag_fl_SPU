# V20 当前进展实验分析：从 M0-Confirm 到 U1 因子化审计

**日期：** 2026-08-04  
**当前主结论：** `routing_primary_bottleneck_confirmed`  
**Reader 状态：** `blocked_before_reader`

## 1. 问题与阶段演进

V20 研究的是固定通信预算下的联邦多跳检索：每个查询只能联系三个客户端（`Bc=3`），每个客户端最多上传五篇文档，总传输预算为 15，服务器输出 top-10 context。前期 HotpotQA 的结果表明，跨客户端直接比较本地 hybrid 分数会造成明显的后处理排序损失；rank-percentile 合并可以恢复其中一部分证据。

然而，M0-Confirm 的跨数据集回放没有通过预注册 gate：2Wiki 的 percentile 改善为 +4.0pp，低于 5pp 阈值；MuSiQue 几乎无净改善且有 harm。因此，不能把 Hotpot 的 merge 校准写成已泛化的方法，也不能直接将失败简化为“只需要训练新 router”。2Wiki/MuSiQue 可能同时受 route coverage 和 local retrieval 影响。

U1 的作用正是将这两个来源正交分解。它固定 partition、查询样本、retriever、`Bc=3`、local depth=10、15-document budget 和 top-10 output；唯一的 Oracle 是离线审计中至多三个 gold-support 客户端的选择，不进入任何可部署路径。

## 2. 共同冻结合同

| 项目 | 设置 |
|---|---|
| 数据 | 2WikiMultiHopQA / MuSiQue development-only，各冻结 N=300 |
| 分区与样本 | 沿用 M0-Confirm，未修改 partition 或 query IDs |
| 实际路由 R0 | inherited origin-plus-topic-centroid router，固定 `Bc=3` |
| 审计路由 R1 | Oracle Bc=3，最多三个 gold-support clients，仅 offline audit |
| 本地排序器 | L0 BGE dense，L1 BM25，L2 `0.55 dense + 0.45 sparse`，L3 RRF |
| 候选合同 | 四个 ranker 共用每客户端 BM25 top-100 candidate pool，再各取 local top-10 |
| 合并检查 | A0 5/5/5 transmission；raw、rank-percentile、RRF server merge |
| 禁止项 | router/retriever/calibrator 训练、reader、final test、gold 推理特征 |
| 可复现性 | 每个数据集 run1/run2 的矩阵与逐 query 输出 byte-identical |

Gold 只在候选构造完成后用于 Oracle coverage、支持文档指标和误差统计。候选物化脚本显式记录 `gold_or_answer_fields_used=false`。

## 3. M0-Confirm 回顾：校准信号为何不足以启动 Reader

| 数据集 | coverage@3 | local complete@10 | A0 raw@10 | A0 percentile@10 | Delta | Rescue / Harm |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.707 | 0.487 | 0.300 | 0.423 | **+0.123** | 37 / 0 |
| 2Wiki | 0.400 | 0.170 | 0.117 | 0.157 | +0.040 | 12 / 0 |
| MuSiQue | 0.423 | 0.190 | 0.120 | 0.123 | +0.003 | 4 / 3 |

HotpotQA 验证了一个真实的后处理问题：已经传出的两跳支持文档会被 raw merge 丢弃，而 rank-based normalization 可以恢复 37 条查询。2Wiki 的方向一致，但改善未达到预注册阈值；MuSiQue 的传输前 local-depth 机会虽增加，却没有转化为可靠的 merge 改善。

因此 M0 的合格结论只是：**rank-percentile 是 Hotpot 条件下有效、在 2Wiki 上有弱复现的无标签后处理机制**。它不是跨域主方法，reader 也不应在此时启动。

## 4. U1 Oracle Routing Audit：路由覆盖是主要可恢复缺口

U1 将 R0 与 R1 放在相同本地 dense ranker L0 下比较。R1 不改变 local ranking，只将最多三个已知支持客户端作为审计 route，因而其增益是“当前 Bc=3 路由本可恢复的上界”，不是可部署性能。

| 数据集 | R0 actual coverage@3 | R1 Oracle coverage@3 | R0L0 local@10 | R1L0 local@10 | RoutingGain | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.400 (120/300) | 0.977 (293/300) | 0.183 | 0.387 | **+0.203** | [+0.160, +0.250] |
| MuSiQue | 0.423 (127/300) | 0.983 (295/300) | 0.177 | 0.407 | **+0.230** | [+0.183, +0.280] |

两个数据集的 Oracle coverage 都接近 98%，说明大多数多跳问题在 `Bc=3` 这一预算下原则上是可覆盖的；当前 inherited route 只覆盖约 40%--42%。因此不是 client budget 太小，而是现有 origin-plus-centroid route 没有稳定找全分散在多个 client 的支持源。

## 5. Frozen Local Retrieval Audit：有局部差异，但非主导瓶颈

### 2Wiki

R0 实际 route 下，dense local@10 为 `0.183`，BM25 为 `0.150`，固定 hybrid / RRF 都为 `0.170`。按预注册定义，LocalGain 比较 L1--L3 相对 L0 的最佳固定选择，结果为 `-0.013`，95% CI `[-0.030, +0.000]`。在已经 coverage 的 120 条查询上，dense conditional complete@10 为 `0.458`，高于 BM25 `0.375` 与 hybrid `0.425`。

这意味着 2Wiki 不是“BGE + sparse 的固定混合不够强”导致的主问题。稀疏证据对个别查询有补充，但平均上不能超越 dense；训练 query-adaptive hybrid 没有足够的上游依据。

### MuSiQue

R0 下 dense local@10 为 `0.177`，hybrid 为 `0.190`，RRF 为 `0.183`，BM25 为 `0.140`。最佳固定 local variant 是 hybrid，LocalGain 仅 `+0.013`，95% CI `[-0.010, +0.037]`，远小于 RoutingGain。在 actual-covered 127 条查询上，hybrid conditional complete@10=`0.449`，dense=`0.417`，显示 MuSiQue 确有可解释的 lexical 补充，但总体幅度不足以作为下一主线。

## 6. 联合作用、传输与合并

| 数据集 | JointGain（R1 + 最佳固定 L） | 95% CI | Interaction |
|---|---:|---:|---:|
| 2Wiki | +0.187（R1 + hybrid） | [+0.140, +0.237] | -0.003 |
| MuSiQue | +0.247（R1 + hybrid） | [+0.193, +0.303] | +0.003 |

交互项接近零，表明 route coverage 与 frozen local retrieval 的贡献近似可加：首先把正确客户端纳入 route，才有机会观察局部 ranker 的小幅差异。没有证据显示二者需要同时训练才能产生超加性收益。

传输与 merge 审计也没有使 merge 再次成为主问题：U1 中任一 R×L 条件的 `transmitted_complete@15 - raw_complete@10` 没有稳定超过 5pp。特别是 MuSiQue R1L0 的 local@10=`0.407` 到 transmitted=`0.357` 的下降主要是固定 5/5/5 transmission 的 allocation 损失，而非 raw server merge。故当前不应回到 learned calibrator 或复杂 merge 模块。

## 7. 当前可支持与不可支持的学术结论

**可支持：**

1. 在两个独立多跳数据集上，固定 `Bc=3` 路由漏掉支持客户端是最大的可恢复上游损失，Oracle audit 给出 +20pp 级别且 bootstrap CI 不跨零的空间。
2. 该发现不依赖 reader、不依赖 retriever 训练，也不依赖对 development 的逐 query ranker 选择。
3. local dense/BM25/hybrid/RRF 的比较解释了 M0 的跨域不一致：当 route 覆盖低时，后处理校准无法把未进入候选池的证据带回来。
4. Hotpot rank-percentile 结果应保留为条件性 merge-mechanism evidence，而不是 V20 的统一主结果。

**不可支持：**

1. R1 Oracle 不可部署，不能报告为新方法性能，不能进入 reader/QA 主表。
2. 尚不能声称 proposed recall-first router 会达到 Oracle 增益；它仍需在新的独立冻结开发协议中验证。
3. 尚未证明 reader 或最终 Answer/Joint F1 改善；reader 正确地处于 blocked 状态。

## 8. 下一步方法决策

最终状态为：

> **`routing_primary_bottleneck_confirmed`**

下一模块应仅开发一个 **multi-prototype recall-first router**，目标是在不增大 `Bc=3`、不改变 partition、不训练 retriever 的前提下，提高 support-client coverage。推荐的下一步设计要求是：

1. 在独立、重新冻结的 development protocol 上训练/选择 router；不能在本 U1 的 300 条上调参；
2. 用 query 的多个语义原型或实体/bridge 表示召回多个 client，而不只依赖 origin + 单 centroid 邻居；
3. 预先声明 coverage@3、local complete@10、15-document merged complete@10 为主检索指标；
4. 先在 2Wiki 与 MuSiQue 上做 retrieval-only 验证。只有两数据集均显示稳定 coverage 和 complete-support 改善，才允许冻结后启动双 reader。

## 9. 关键产物

- U1 2Wiki：`stage_u1_factorized_audit/2wikimultihopqa/run1/`
- U1 MuSiQue：`stage_u1_factorized_audit/musique/run1/`
- Oracle coverage：各数据集 `oracle_routing/oracle_coverage.csv`
- R×L 主矩阵：各数据集 `factorial_matrix/routing_local_matrix.csv`
- 条件性 dense/BM25 分析：各数据集 `error_analysis/dense_bm25_complementarity.csv`
- U1 决策：各数据集 `reports/next_method_decision.json`
- 实现提交：GitHub commit `0cba1c9`。
