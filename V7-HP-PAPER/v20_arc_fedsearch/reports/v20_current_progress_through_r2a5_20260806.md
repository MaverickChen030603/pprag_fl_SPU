# V20 当前进展实验分析报告：从 Merge Audit 到 Candidate Compression

**报告日期：** 2026-08-06  
**当前可部署方法状态：** 尚未成立  
**当前诊断结论：** `routing_primary_bottleneck_confirmed`，且首版 candidate representation 存在瓶颈  
**Reader / Final Test：** 均未运行、保持封存

## 1. 研究目标与当前位置

V20 的目标是在联邦多跳检索中，在文档和完整索引不离开客户端、每个查询最多联系三个客户端（`Bc=3`）、每个客户端上传五篇文档、总通信预算固定为 15 篇的条件下，提高完整 supporting evidence 进入服务器 top-10 context 的概率。

当前实验并未把检索收益提前归因于 reader。所有结果均是 retrieval-only：gold support 只用于训练标签（尚未启动 router 训练）、离线 Oracle 或最终指标；它不进入任何 route、profile、candidate、selector 或 local retrieval 的推理特征。这样做的好处是当前结论虽然还不是最终 QA 论文结果，但其失败位置是可定位的。

实验链条已完成三个阶段：

1. **M0-Confirm：** 固定 Bc=3 下的 local-depth、15-document transmission 与跨客户端 merge 审计；
2. **U1：** 将 route coverage 与 frozen local retrieval quality 正交分离；
3. **R2-A 与 R2-A.5：** 评估首版 multi-prototype resource profile 的候选召回，以及从 candidate Top-5 压缩到 Bc=3 时是否存在可由简单 set-aware 规则恢复的互补 client。

## 2. M0-Confirm：rank calibration 是条件性机制，而非主方法

| 数据集 | client coverage@3 | local complete@10 | raw merged@10 | rank-percentile@10 | Delta | Rescue / Harm |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.707 | 0.487 | 0.300 | 0.423 | **+0.123** | 37 / 0 |
| 2Wiki | 0.400 | 0.170 | 0.117 | 0.157 | +0.040 | 12 / 0 |
| MuSiQue | 0.423 | 0.190 | 0.120 | 0.123 | +0.003 | 4 / 3 |

HotpotQA 清楚地证明了：在支持文档已由 selected clients 上传的条件下，直接比较 client-local hybrid score 会损失证据，rank-percentile 可以恢复这一后处理损失。然而，这种改善没有以预注册强度跨数据集复现：2Wiki 的 +4pp 低于 +5pp 门槛，MuSiQue 几乎为零且出现 harm。

因此，论文中若保留该部分，应将其写作“**cross-client score calibration 的条件性诊断**”，不能写作统一的联邦检索方法成功，更不能把 Hotpot 的结果外推为 reader 或 QA 收益。

## 3. U1：路由覆盖是两个数据集上最大的可恢复损失

U1 固定相同 local dense top-100 candidate contract，对比实际 inherited origin-plus-topic-centroid route（R0）与最多三个 gold-support clients 的 offline Oracle route（R1）。R1 只量化机会，不是部署结果。

| 数据集 | R0 coverage@3 | R1 Oracle coverage@3 | R0 dense local@10 | R1 dense local@10 | RoutingGain | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.400 | 0.977 | 0.183 | 0.387 | **+0.203** | [+0.160, +0.250] |
| MuSiQue | 0.423 | 0.983 | 0.177 | 0.407 | **+0.230** | [+0.183, +0.280] |

在同一合同下，最佳冻结 local ranker 对 dense 的收益只有 2Wiki `-0.013`、MuSiQue `+0.013`，CI 均跨零；route-local interaction 也接近零。这个结果排除了“只换 BM25/hybrid 就会解决问题”的解释，并说明在 `Bc=3` 预算内大多数查询原则上可被覆盖，当前 route 却没有稳定找到多个 hop 所在客户端。

边界同样重要：U1 只适用于共享 BM25 top-100 candidate pool 下的 local retrieval 合同，不能声称已排除 full-index dense retrieval 的所有局限。

## 4. R2-A：首版 MARS-Route resource cards 未通过候选门槛

为避免使用 U1 analysis set 调方法，R2 重新冻结了 Router-Train 5000、Router-Dev 300、Router-Calibration 200 和 fresh Router-Holdout 500。fresh holdout 与 U1 `development[100:400]` 零重叠，final test 未读取。

R2-A 在 Router-Dev 前 100 条比较：P0 single centroid、P1 all-document multi-prototype profile（P=4/8/16）、P2 lexical sketch、P3 dense+lexical RRF。候选打分只使用 query 和 client resource card。

| 数据集 | P0 complete-set recall@5 | 最佳 P1 Q0 | Delta | P3 dense+lexical RRF | Delta |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.720 | P8: 0.750 | +0.030 | 0.630 | -0.090 |
| MuSiQue | 0.620 | P16: 0.660 | +0.040 | 0.700 | +0.080 |

预注册要求两个数据集相对 P0 至少 +5pp。P1 只有小幅、不一致收益；deterministic entity/clause/relation multiview 在两个数据集均严重退化；lexical fusion 只帮助 MuSiQue 而伤害 2Wiki。因此未进入 R2-B set-aware selector，也没有调用 ReSLLM teacher 或训练 student。

## 5. R2-A.5：候选压缩机会存在，但当前无训练 selector 无法利用

R2-A.5 固定复用 R2-A 的 profile、Q0、Router-Dev N=100、local dense retriever、A0 和 Bc=3。它的目的不是提出方法，而是区分两种失败：

- **Candidate absence：** 完整 gold client set 连 Top-8 都未进入；
- **Selection compression：** gold clients 在 Top-5 中已齐全，却被独立 Top-3 丢弃。

### 5.1 P0 candidate ladder

| 数据集 | P0 complete-set@3 | @5 | @8 | OracleSubset@3 within Top-5 | within Top-8 | CompressionGap@5 | CandidateAbsenceLoss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.540 | 0.720 | 0.890 | 0.720 | 0.890 | **0.180** | 0.100 |
| MuSiQue | 0.460 | 0.620 | 0.800 | 0.620 | 0.800 | **0.160** | 0.200 |

结果揭示了一个混合事实：Top-5 中确实有相当的集合压缩机会（16--18pp），但还有 10pp（2Wiki）与 20pp（MuSiQue）的 Oracle-coverable query 连 P0 Top-8 都没有完整 client set。也就是说，当前数据不能被简化为“仅需更聪明的 Top-3 selector”。

### 5.2 无训练 selector probes

S0 为 independent Top-3；S1 为三档固定 lambda 的 redundancy-penalized greedy；S2 为 prototype-specialization greedy；S3 仅在 dense/lexical 明显不一致时插入 lexical-exclusive client。

| 数据集 | S0 coverage@3 | 最佳 S1/S2 coverage@3 | Gain / 95% CI | S0 local complete@10 | 最佳 probe local complete@10 |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.540 | 0.540 | +0.000, [-0.140, +0.140] | 0.200 | 0.200 |
| MuSiQue | 0.460 | 0.460 | +0.000, [-0.130, +0.140] | 0.250 | 0.250 |

S3 反而退化：2Wiki coverage 为 0.460，MuSiQue 为 0.410。所有 probe 的 transmitted 与 raw/percentile merged complete 均未改善。2Wiki 的 compression-failure query 的 top-3 profile cosine 比成功 query 高 `+0.025`，95% CI `[+0.013,+0.035]`；MuSiQue 的差异为 `-0.001`，CI 跨零。故“高冗余”并不是两个数据集均可用的 selector 信号。

## 6. 当前总判断

当前不能直接开发 learned set selector，原因有三：

1. 两个数据集都有显著的 candidate absence，尤其 MuSiQue Top-8 的完整集合缺失为 20pp；
2. 允许的简单 diversity/prototype probes 没有将已有的 compression opportunity 转化为 coverage 或 document evidence；
3. R2-A 的 resource representation 本身没有在两数据集稳定提高 candidate recall。

因而 R2-A.5 的最终状态为：

> **`candidate_representation_bottleneck_confirmed`**

它不否定 U1 的 routing opportunity，而是把下一问题精确化：需要先改善“候选 client 资源表示与 query-resource matching”，再讨论预算集合选择。否则训练 selector 很可能只会在已有差候选中重新排序，并把开发集噪声当作方法增益。

## 7. 对 9 月投稿的建议

当前最重要的是止损与选择论文叙事，而不是继续堆叠 router、LLM teacher 和 reader：

### 路线 A：一次严格的 representation recovery gate

仅允许执行指导意见中定义的 **Representative Evidence Memory Profile**：从 train corpus 构建有限代表单元（diverse titles、实体、short snippets、rare discriminative units），不再添加 deterministic multiview，不重训 profile clustering，也不在当前 Dev100 上调 set weights。新表示必须在新的预注册 Router-Dev slice 上同时使 2Wiki、MuSiQue candidate complete-set recall@5 相对 P0 提升至少 +5pp。

通过后才进入 R2-B 的 fixed-Bc=3 selector；失败则停止 MARS-Route 方法线。

### 路线 B：诊断与协议型硕士论文

若 recovery gate 失败，建议将论文收敛为 **Federated Multi-hop Retrieval Bottleneck Audit**：贡献不再是未证实的 MARS-Route，而是严格 no-leak 下的四段损失分解、Oracle opportunity 分析、跨数据集 merge 条件性、candidate absence 与 compression 的区分，以及负结果对 set-aware routing 前提的约束。它需要在写作中明确其诊断性质，不将 Oracle 或 Hotpot-only signal 包装为端到端方法性能。

### 当前禁止事项

在上述 gate 前，禁止 reader、final test、ReSLLM teacher/student、learned set selector、动态 Bc、local retriever/partition 改动。这样仍可在 9 月前形成可信可辩护的成果，而不会因为后期无效扩展破坏 fresh holdout 的价值。

## 8. 核心产物

- M0/U1/R2-A 综合报告：`reports/v20_current_progress_m0_u1_r2a_analysis_20260804.md`
- R2-A.5 per-query 与汇总：`stage_r2a5_candidate_compression/<dataset>/`
- Compression 决策：`stage_r2a5_candidate_compression/<dataset>/reports/r2a5_compression_go_no_go.md`
- R2 split manifest：`stage_r2_mars_route/<dataset>/protocol/router_split_manifest.json`
- 当前实现提交：`e24fba9`、`cef6cdf`。
