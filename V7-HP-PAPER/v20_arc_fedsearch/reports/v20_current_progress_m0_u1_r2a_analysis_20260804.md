# V20 当前进展分析报告：M0、U1 与 MARS-Route R2-A

**日期：** 2026-08-04  
**当前研究状态：** `resource_representation_failure`（针对首版 MARS-Route resource cards）  
**仍成立的上游诊断：** `routing_primary_bottleneck_confirmed`  
**Reader：** `blocked_before_reader`

## 一、研究目标与证据策略

本项目的最终目标是形成一篇可投稿的联邦多跳检索论文：在文档与完整索引保持客户端本地、每条查询最多联系三个客户端、总上传文档数固定为 15 的条件下，改善完整两跳支持证据进入服务器 context 的概率。

为避免把检索、路由、合并与 reader 混为一谈，V20 采用逐级门控：M0 检验后处理 merge，U1 正交分解 route 与 local retrieval，R2-A 才检验新的 resource representation 是否能为可部署 router 提供足够候选召回。当前没有启动 reader，因此不存在任何 Answer F1、Joint F1 或 reader feedback 的正向宣称。

## 二、M0-Confirm：跨客户端 raw merge 是真实但非普适的后处理问题

| 数据集 | coverage@3 | local complete@10 | raw merged@10 | rank-percentile@10 | Delta | Rescue/Harm |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.707 | 0.487 | 0.300 | 0.423 | **+0.123** | 37 / 0 |
| 2Wiki | 0.400 | 0.170 | 0.117 | 0.157 | +0.040 | 12 / 0 |
| MuSiQue | 0.423 | 0.190 | 0.120 | 0.123 | +0.003 | 4 / 3 |

HotpotQA 上，rank-percentile 能恢复 raw cross-client score merge 丢失的文档；该结果在双重复运行中逐 query 字节一致。2Wiki 的改善方向相同但低于预注册 +5pp 门槛；MuSiQue 没有稳定净收益。因此不能将 percentile 表述为跨数据集已验证的方法，也不能据此启动 reader。

M0 的正确定位是条件性机制证据：**当支持客户端已经被 route 覆盖、且证据已经传输时，跨 client 的原始局部分数尺度可能不可比较。**它不解决未被联系 client 的证据缺失。

## 三、U1 因子化审计：路由而非冻结 local ranker 是主要可恢复缺口

U1 使用同一 local dense contract，对比实际 inherited Bc=3 route（R0）与最多三个 gold-support client 的 offline Oracle route（R1）。Oracle 仅用于诊断，绝不是部署结果。

| 数据集 | actual coverage@3 | oracle coverage@3 | R0 dense local@10 | R1 dense local@10 | RoutingGain | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | 0.400 | 0.977 | 0.183 | 0.387 | **+0.203** | [+0.160, +0.250] |
| MuSiQue | 0.423 | 0.983 | 0.177 | 0.407 | **+0.230** | [+0.183, +0.280] |

冻结 local ranker 的最佳固定替代相对 dense 只有：2Wiki `-0.013`，MuSiQue `+0.013`，且 CI 均跨零。route-local interaction 约为零。该证据支持以下结论：在当前 partition 和 `Bc=3` 预算下，大量支持 client 理论上可以被三 client 覆盖，但 inherited origin-plus-centroid route 没有找全多跳来源。

这并不等同于“已排除全部 full-index dense retrieval 问题”。U1 的结论严格限定于共享 BM25 top-100 candidate pool 与冻结 local ranker 合同；它足以决定下一模块优先改善 route coverage，但不能替代未来的 full-index audit。

## 四、R2-A：首版多原型资源表示未通过候选召回门槛

R2 为避免在 U1 的 analysis set 上调参，重新冻结了：Router-Train 5000、Router-Dev 300、Router-Calibration 200、fresh Router-Holdout 500。Holdout 与 U1 development[100:400] 零重叠，final test 未访问。

R2-A 在 Router-Dev 前 100 条上从全部 client-local 文档构建 P0 single centroid、P1 multi-prototype（4/8/16）、P2 lexical sketch 和 P3 dense+lexical RRF resource cards。候选生成只用 query 与 profile；gold client 只在生成后离线计 recall。

| 数据集 | P0 Q0 complete-set recall@5 | 最佳 P1 Q0 | P1 Delta | P3 P8 dense+lexical RRF | P3 Delta |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.720 | P8: 0.750 | +0.030 | 0.630 | -0.090 |
| MuSiQue | 0.620 | P16: 0.660 | +0.040 | 0.700 | +0.080 |

预注册门槛是相对 P0 的 complete-client-set recall@5 或 gold-client recall@5 至少 +5pp。没有一种表示同时在两个数据集达到这一标准：P1 Q0 有小幅、但不一致的改进；deterministic entity/clause/relation multiview 在两个数据集均明显退化；lexical fusion 仅对 MuSiQue 有益且损害 2Wiki。

因此当前 R2-A 的正式结论是 `resource_representation_failure`。这是首版 resource-card / query-view 设计失败，而不是对 U1 路由诊断的否定：我们知道 route coverage 有大上限，但尚未得到一个足够强、可部署的高召回候选生成器。

## 五、方法取舍与论文影响

### 已有可写价值

1. **可审计的损失分解。** M0 与 U1 共同将联邦多跳失败区分为 local depth、15-document transmission、cross-client merge、route coverage 四类，并以冻结 Bc=3 合同测量。
2. **强的 Oracle opportunity 证据。** 两数据集的 route upper gap 均约 20pp，paired bootstrap CI 明确为正；这比仅报告小幅下游提升更能解释问题结构。
3. **有价值的负结果。** 简单 multi-prototype、规则 multiview 和 lexical fusion 未稳定提高 candidate complete-set recall。该结果能防止论文将 resource-card 的复杂性误写为贡献。
4. **严谨边界。** 无 reader、无 final-test、所有 M0/U1 重复矩阵 byte-identical；Oracle 与可部署结果分开报告。

### 尚不具备的投稿主张

1. 尚无可部署 router 相对 inherited route 的 fresh-holdout 改善；不能声称 MARS-Route 有效。
2. 尚无 reader-backed QA 收益；不能以 RAG answer/joint performance 为主结果。
3. 尚无 ReSLLM teacher 或 student 价值；不应因为名称新颖而加入 LLM teacher。
4. Hotpot 的 percentile 信号不能包装成跨数据集统一胜利。

## 六、对 9 月投稿的建议

当前最稳妥的时间策略是设置一个短而硬的 router representation recovery gate：

1. 仅做 profile-quality audit：prototype size/variance、client-profile overlap、错误 client 与正 client 的 title/entity 重叠，以及 multiview 的错误来源；不训练 selector。
2. 基于该审计提出**一项**预注册的 representation 修订，例如 query view filtering 或更高保真 entity sketch，而不是同时加入 learned router、set objective 与 LLM teacher。
3. 在新的、未使用 Router-Dev N=100 上复跑 candidate smoke。若两数据集仍不能达到 +5pp，则停止 MARS-Route 方法线，避免在 9 月前消耗时间于低概率堆叠。
4. 若通过，才进入 R2-B：固定 Bc=3，比较 independent top-3 与 greedy set-aware selection；再由 fresh holdout、多 seed 与 paired bootstrap 决定是否运行 reader。

若 recovery gate 不通过，论文仍可转向更可信的“Federated Multi-hop Retrieval Bottleneck Audit”定位：强调 route coverage opportunity、merge calibration 条件性、候选表示失败和严格 no-leak protocol。该定位需要补充更广的 partition/profile audit 与相关工作，但比将未证实的 MARS-Route 写成方法成功更符合硕士成果论文的可辩护性。

## 七、当前决策

> **禁止 R2-B、teacher/student、reader。**  
> **允许的唯一下一步：profile-quality audit 与一次独立、预注册的 candidate-representation recovery smoke。**

## 八、产物

- M0/U1 报告：`reports/v20_current_progress_m0_u1_factorized_analysis_20260804.md`
- R2-A no-go：`stage_r2_mars_route/reports/r2a_candidate_smoke_go_no_go_20260804.md`
- R2 split manifest：各数据集 `stage_r2_mars_route/<dataset>/protocol/router_split_manifest.json`
- R2 profiles：各数据集 `resource_profiles/client_profiles.json` 与 `prototype_statistics.csv`
- R2-A candidate recall：各数据集 `candidate_generation/r2a_smoke100_v2/candidate_recall.csv`
- 实现提交：`97c58ce`、`a736187`、`9fca2ad`。
