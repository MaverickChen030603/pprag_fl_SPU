# V7-HP-PAPER V15 阶段实验报告

更新日期：2026-07-21  
当前状态：`needs_more_experiments`  
工作名：Robust Risk-Calibrated Context Repair

## 1. 执行摘要

V15 的目标不是继续修饰 V14 的论文叙述，而是重做三处关键机制：扩大完整
context repair 的可用机会、直接对齐官方 Reader 指标、以及用逐 query 风险决策
替换已知 batch 上的全局 Top-B。当前已完成新协议、全新 train-derived 数据冻结、
真实 Top-10/Top-20 检索池基础设施、enumerated/beam repair、双 Reader 结果标注、
direct-delta scorer 与风险 gate 的第一版实现，并跑完 HotpotQA 100-query pilot。

阶段结果表明：

1. **候选机会真实存在。** 在每题只评估 16 个 repair 的保守 pilot 中，FLAN 与
   UnifiedQA 的 Joint-oracle 分别比各自 baseline 高 `+0.0528` 和 `+0.0474`。
2. **机会尚未稳定转化为学习式收益。** MLP 在 FLAN 上能排序 Joint delta，
   但在 UnifiedQA 上相关性为负；分别最大化单 Reader 会伤害另一 Reader。
3. **稳健目标出现小正信号。** 预注册网格中的 `beta=1` 在 22 条 held-out
   development query 上实现平均 Reader Joint `+0.0113`、最差 Reader 平均
   `+0.0022`，且未观察到 Reader harm。该结果样本小，只能支持继续实验。
4. **暂不解封 final test。** 当前最合理的下一步是增加 train-derived 标注 query，
   直接训练 robust mean/min utility，再在独立 calibration 上冻结 gate。

因此，Checkpoint 1 判为“部分通过”，不能宣称 method breakthrough。

## 2. V14 诊断与 V15 研究问题

V14 的核心限制包括：动作空间太小导致 training-positive repair 缺失；旧 utility
使用 Answer 与 title-F1 代理 Joint；全局 Top-B 依赖已知 batch；单 Reader 监督
缺乏迁移稳定性；CrossEncoder 在 Joint 与延迟上仍更强。V15 围绕以下问题展开：

- 真实 Top-L pool 中，完整 sequence repair 是否减少 search-level absence？
- 直接预测 Answer/SP/Joint delta 是否优于旧 title proxy？
- 同一 action 能否同时服务两套冻结 Reader？
- 独立逐 query gate 能否在控制 Answer/Joint harm 的同时保留增益？
- cheap-to-expensive cascade 能否把 post-retrieval 成本压到可投稿范围？

## 3. 数据协议与泄漏控制

### 3.1 历史使用盘点

协议脚本扫描了 1,563 个历史结构化 artifact，建立 ID 与规范化 question
fingerprint 双重排除。库存包含：

| 数据集 | 历史 ID | 历史规范化 question |
|---|---:|---:|
| HotpotQA | 8,405 | 12,405 |
| 2WikiMultiHopQA | 2,187 | 2,187 |

对缺少 query ID 的历史文件采用 question fingerprint 保守排除。五个解析错误已
记录，其中四个为 macOS metadata，一个为历史损坏 JSONL。

### 3.2 新冻结数据

数据来自此前未用于方法选择的官方 train source：HotpotQA 90,447 条、2Wiki
167,454 条。每个数据集固定：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 5,000 | scorer、support predictor、cheap gate |
| development | 1,000 | 模型/预注册网格一次性选择 |
| calibration | 1,000 | 风险阈值独立校准 |
| final frozen test | 2,000 | 方法冻结后的 confirmatory evaluation |

HotpotQA 排除 1,000 个历史 ID 与 5,002 个历史 question match；2Wiki 排除
638 个 train question match。final 输入与标签物理分离，标签文件权限为 `0400`。
截至本报告生成，V15 未读取 sealed final label。

### 3.3 冻结合同

- Pool：真实 Top-10 与 Top-20；无随机文档、无 gold padding。
- Retriever：BGE dense + BM25 sparse，固定 `alpha=0.55`。
- Reranker：`cross-encoder/ms-marco-MiniLM-L-6-v2`。
- Reader A：`google/flan-t5-large`。
- Reader B：`allenai/unifiedqa-v2-t5-large-1363200`。
- 最终 context budget：5 documents。
- Primary：Answer F1、SP F1、Joint F1、Answer/Joint harm、coverage、latency。
- 统计：query-paired bootstrap 5,000 次，95% CI，secondary comparison 做 BH。

## 4. 方法设计

### 4.1 真实候选池

已建立两个训练语料 FTS5 索引：HotpotQA 481,959 文档、2Wiki 369,280 文档。
检索先生成全局真实候选，再计算 dense/sparse hybrid 分数和 CrossEncoder 分数。
benchmark 自带 context 仅作为合法 corpus 文档进入索引，不以 gold 身份插入结果。

### 4.2 Complete-Sequence Repair Search

当 `L<=12` 时，枚举五文档子集，并生成 baseline-preserving、retrieval-score、
CrossEncoder 与 bridge-first 顺序；cheap inference-safe scorer 保留 K 个动作。

当 `L>12` 时，从 Top-5 出发执行 replace、insert-and-remove、reorder，使用 beam
width/depth、duplicate pruning 与 upper-bound pruning。baseline 永远保留为 exact
null action。Hotpot Top-20 smoke 每题暴露 63 个非空动作与一个 baseline。

### 4.3 Direct Multi-Objective Scorer

第一阶段采用 MLP/GBDT，而非直接升级重型 Transformer。输入是 50 个推理安全
sequence 特征，包括 hybrid/dense/sparse/CrossEncoder 统计、added/removed 文档、
anchor preservation、order agreement、query overlap、pair diversity、title bridge
和长度。禁止 `is_support`、answer presence、gold outcome 等标签特征。

模型对每个 Reader 同时预测：

- Delta Answer F1；
- Delta SP F1；
- Delta Joint F1；
- P(Answer drop)；
- P(Joint drop)。

loss 由 Smooth-L1 regression、BCE harm classification 与 within-query hinge ranking
组成。训练日志显示梯度范数约 0.20--0.24，没有梯度爆炸。

### 4.4 Multi-Reader Robust Utility

同一 action 保留 Reader-specific label。稳健效用使用预注册形式：

`U_beta = mean_r(Delta Joint_r) - beta * std_r(Delta Joint_r)`，

其中 `beta in {0, 0.25, 0.5, 1.0}`，最终值只能在 development 选择一次。
严格评估要求同一 action 同时作用于两套 Reader，不能分别挑各自 oracle。

### 4.5 Per-Query Risk Gate

经验 gate 对每题独立判断 best predicted utility 与预测 harm，未通过则 exact
fallback。形式化版本已实现有限阈值族、Bonferroni 修正、二项风险上界和均值
下界，但在 assumptions audit 与 calibration 实验完成前，只能称为
`empirically calibrated per-query risk gate`，不能声称分布无关保证。

### 4.6 Cost-Aware Cascade

Stage 1 仅使用 BM25、rank、lexical/entity overlap、cached dense 与冻结 baseline
特征；只有通过或不确定的 query 才进入 repair/scorer。`07_train_cheap_gate.py` 与
`08_cost_aware_inference.py` 已实现并完成 pilot 跑通，但当前 78-query train / 22-query
development 规模下，为达到 0.90 opportunity recall，只能取 gate threshold=0，
expensive-stage invocation 为 100%。因此 cascade 目前没有实现成本节省，也不能
声称达到 170 ms/query 目标。

## 5. Retrieval 与 Repair Smoke

### 5.1 真实检索池

| 数据集 | Pool | Support recall | Complete support | Answer document | Mean retrieval latency |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 10 | 0.870 | 0.740 | 0.900 | 1331 ms |
| HotpotQA | 20 | 0.900 | 0.800 | 0.920 | 1331 ms |
| 2Wiki | 10 | 0.680 | 0.360 | 0.600 | 1139 ms |
| 2Wiki | 20 | 0.710 | 0.400 | 0.660 | 1139 ms |

Top-20 在两个数据集都增加检索机会，但 2Wiki complete-support 只有 0.40，说明
第二数据集的主要风险位于上游 pool-level absence，不能全部归因于 selector。
这里的检索延迟是上游 retrieval，不与 post-retrieval action latency 混报。

### 5.2 Repair 搜索开销

Hotpot Top-20 beam 产生 64 个去重动作/题，其中包含 exact fallback；平均生成耗时
84.6 ms，最大 95.6 ms。100-query Reader pilot 为控制标注成本，只对每题 16 个
动作运行两套 Reader，而不是把所有 64 个动作全部标注。

## 6. HotpotQA 100-Query 双 Reader Pilot

### 6.1 Action-set opportunity

| Reader | Baseline Answer | Baseline SP | Baseline Joint | Joint-oracle Answer delta | SP delta | Joint delta | 有正 Joint 机会的 query |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLAN | 0.6553 | 0.4457 | 0.3309 | +0.0775 | +0.0228 | +0.0528 | 19% |
| UnifiedQA | 0.4955 | 0.4457 | 0.2627 | +0.0753 | +0.0110 | +0.0474 | 15% |

非 baseline 动作导致 Joint 下降的比例分别为 14.53% 和 12.00%。这同时说明 repair
pool 具有正机会，也说明无 gate 的全量干预并不安全。

### 6.2 Same-action robust oracle

| Beta | Oracle intervention | Mean-reader Joint delta | Min-reader Joint delta | 两 Reader 同时正向 | 任一 Reader harm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.230 | +0.0481 | +0.0156 | 0.110 | 0.010 |
| 0.25 | 0.220 | +0.0478 | +0.0178 | 0.110 | 0.000 |
| 0.5 | 0.220 | +0.0475 | +0.0185 | 0.110 | 0.000 |
| 1.0 | 0.110 | +0.0229 | +0.0185 | 0.110 | 0.000 |

这些结果证明 action set 中存在可同时服务两套 Reader 的 repair，但它们使用真实
outcome 选动作，属于 retrospective upper bound，不能写入主表作为方法收益。

### 6.3 Direct scorer

22-query held-out development 子集上的主要结果：

| Reader | Target | Spearman | Pairwise accuracy | Predicted-top realized delta |
|---|---|---:|---:|---:|
| FLAN | Answer | 0.1405 | 0.6255 | +0.0000 |
| FLAN | SP | 0.1676 | 0.3846 | -0.0032 |
| FLAN | Joint | 0.2184 | 0.7156 | +0.0068 |
| UnifiedQA | Answer | -0.0644 | 0.4313 | -0.0390 |
| UnifiedQA | SP | 0.0460 | 0.5625 | +0.0052 |
| UnifiedQA | Joint | -0.0670 | 0.4729 | -0.0206 |

FLAN Joint-harm AUROC 为 0.7895，UnifiedQA 为 0.6109；说明 harm head 有一定价值，
但 reader-independent action ranking 尚未学稳。GBDT 并未稳定改善两 Reader：其
FLAN/UnifiedQA Joint Spearman 分别为 0.0237/0.1336，仍然一好一坏。

### 6.4 Robust learned diagnostic

在同一 22-query held-out 子集上，使用预测的跨 Reader robust utility 选择同一动作：

| Beta | Intervention | Mean-reader Joint delta | Min-reader Joint delta | 任一 Reader harm |
|---:|---:|---:|---:|---:|
| 0 | 0.5909 | -0.0058 | -0.0365 | 0.0909 |
| 0.25 | 0.5909 | -0.0172 | -0.0547 | 0.0909 |
| 0.5 | 0.5000 | -0.0172 | -0.0547 | 0.0909 |
| 1.0 | 0.4545 | +0.0113 | +0.0022 | 0.0000 |

`beta=1` 的保守目标产生小正信号，但 N=22，且尚未经过独立 calibration 和 final
test。它只支持下一阶段把训练 loss 改为 robust mean/min ranking，并增加 query
数量；不支持宣称 V15 已经解决跨 Reader 泛化。

### 6.5 Cost-aware cascade pilot

cheap gate 使用 15 个 baseline-only 推理安全特征训练，train/development 的正机会率
分别为 10.26%/13.64%。为在 development 保留至少 90% 的机会 recall，阈值退化为
0，调用率为 100%。调用 MLP scorer 时平均前向约 2.07 ms，但这不包括约 84.6 ms 的
repair generation，更不包括上游 retrieval，不能据此宣称端到端降本。retrospective
outcome audit 与 `beta=1` 稳健选择一致：FLAN/UnifiedQA Joint delta 分别为
`+0.0203/+0.0022`，任一 Reader harm 为 0；质量方向可继续，但 cascade 目标失败。

## 7. Checkpoint 判断

### Checkpoint 1：Partial Pass

- 通过项：direct-delta 的平均 Joint 对齐优于旧 proxy；候选 action 对两 Reader
  都有明确 oracle 空间；训练稳定；保守 robust 目标出现 deployable 方向的正信号。
- 未通过项：UnifiedQA 的 direct Joint 排序为负；unconstrained top action 平均不稳；
  只有 22 条 held-out query，无法做显著性与风险保证。
- 决策：保留 MLP/GBDT 路线，不升级重型 Transformer；先扩大 query-level
  reader-labelled training，并直接优化跨 Reader robust ranking。

### Checkpoint 2：Provisional Pass

- Top-20 search 在可控的 84.6 ms 内暴露 64 个动作，并产生真实 Reader-positive
  opportunity。
- 但尚未计算 pool-level/search-level/scoring/gating 的完整层级分解。
- 决策：保留 beam repair；训练时优先增加 query 数而非把所有动作都跑 Reader。

### Checkpoint 3/4：未评估

跨数据集、完整 risk-coverage、总成本、强 baseline 和 untouched final test 尚未完成。

## 8. 风险与当前边界

1. **小样本选择风险：** 22-query robust 正信号可能不稳定。
2. **Reader disagreement：** 两 Reader 的 action delta 只有中等相关，单 Reader
   最优动作不能直接迁移。
3. **2Wiki pool ceiling：** Top-20 complete-support 0.40 可能限制后续 repair 上界。
4. **延迟未闭环：** 已知 repair generation 成本，但 scorer、gate、cascade 的总成本
   未统一测量；第一版 cheap gate 的 expensive-stage invocation 为 100%。
5. **风险保证尚未成立：** formal gate 目前只是经过单元测试的实现 scaffold。
6. **主 baseline 未齐：** CrossEncoder、V14 Full、RECOMP、marginal utility 等尚未
   在新冻结数据和同一 Reader 合同下形成主表。

## 9. 下一步执行顺序

1. 从冻结 train/development 中增加双 Reader 标注 query，保持每题 8--16 个高多样性
   动作，避免先扩到 64 个造成标注成本爆炸。
2. 训练 query-level robust listwise/ranking objective，优化 `mean-beta*std` 与
   minimum-reader utility；beta 仅在 development 选一次。
3. 在独立 calibration 上选择 utility/harm thresholds，输出 risk-coverage curve，
   再决定 formal claim 是否满足 assumptions audit。
4. 完成 2Wiki pilot；若 pool-level absence 主导，先修 retriever，不把失败归咎于 gate。
5. 实现 cheap opportunity gate 与 cost-aware inference，统一记录 post-retrieval
   mean/P50/P95、throughput、GPU memory 和 invocation rate。
6. 运行固定 baselines、十项 ablation 与 failure taxonomy。
7. 所有方法、beta、threshold、reader、pool、seed 冻结后，才解封 2,000+2,000
   final test，并执行 paired bootstrap 与 BH correction。

## 10. 当前论文定位建议

当前最强、且不夸大的论文叙述是：V15 建立了一个从真实候选池到完整 context
repair、官方 Reader delta、跨 Reader robust utility 和逐 query risk gate 的新方法
框架，并在 fresh train-derived pilot 中验证了 action-set opportunity 与保守选择信号。
主方法是否成为非支配 operating point，仍需 calibration、2Wiki、cost 和 final test。

若 Checkpoint 3 达不到至少两项预注册条件，应转为 ECIR analysis paper，重点报告
pool absence、search absence、scoring failure 与 gate failure 的层级分解；不得强行
包装为新 SOTA。若两数据集、两 Reader、风险与成本均成立，再考虑 ARR/ACL 方法稿。

## 11. 主要产物

- `v15_preregistration.md`：预注册协议与决策门槛。
- `protocol/used_query_inventory.md`：历史 query 审计。
- `protocol/data_split_manifest.json`：新冻结 split 来源与数量。
- `protocol/no_leak_audit.json`：标签访问和特征泄漏审计。
- `retrieval/`：真实 corpus 索引与 Top-L pool 生成脚本。
- `action_generation/`：enumerated/beam repair。
- `action_scorer/`：MLP/GBDT direct multi-reader scorer。
- `risk_gate/`：经验 gate 与风险校准 scaffold。
- `cascade/07_train_cheap_gate.py`、`08_cost_aware_inference.py`：两阶段成本原型。
- `analysis/07_robust_utility_diagnostic.py`：learned same-action robust audit。
- `analysis/08_pilot_opportunity_analysis.py`：双 Reader opportunity/oracle audit。
- `results/pilot_hotpot100/`：本地同步的 pilot 指标与训练日志。

本地最新验证为 10/10 单元测试通过；no-leak audit 状态为 `pass`，扫描范围包括
retrieval、action generation、scorer、risk gate、multi-reader、cascade 与 baselines。
