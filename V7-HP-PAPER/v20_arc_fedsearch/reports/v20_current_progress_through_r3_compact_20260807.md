# V20 ARC-FedSearch 当前实验分析报告

**日期：2026-08-07**  
**范围：M0/M1、U1、R2、R2-A.5、R2-A.6 与 R3 ProbeRoute-FedRAG**  
**当前 reader 状态：未启动；final test：未访问**

## 1. 研究问题与当前结论

V20 的目标不是在联邦多跳检索中盲目扩大本地检索深度，而是识别并逐段修复如下链条：

`资源路由 -> 客户端本地检索 -> 固定15-document传输 -> 跨客户端合并 -> reader`。

截至当前，最强且可复现的结论是：**静态 resource profile 难以定位当前 query 的证据客户端，但 query-time 的无正文 local probe 提供了强、跨 2Wiki/MuSiQue 一致的额外判别信号。** 在新的未复用 `Probe-Dev` 上，简单的、无训练 probe 规则已显著提高 3-client complete-set coverage，并将收益传递至 local / transmitted / merged evidence 指标。与此同时，初始“probe 通信过大”的疑问已由固定 float32 wire-format 审计澄清：先前约 7.6 KB 的数值来自 verbose JSON 调试记录，并非正式通信格式；正式 `L=8` probe 为 592 bytes，约为 15-document payload 的 6.7%--7.9%。

这仍不是完整方法成功：当前只完成两个数据集各 100 query 的冻结 Probe-Dev，尚未训练轻量 ranker、尚未做 fresh holdout、未验证 HotpotQA 迁移、更未进行 reader QA。因此，当前可称为 `query_time_probe_signal_confirmed` 与 `compact_probe_communication_contract_confirmed`，不能称为最终的 ProbeRoute 方法确认或端到端 QA 收益确认。

## 2. 冻结合同与可信度边界

当前所有 V20 retrieval-only 阶段保持以下合同：

- 客户端分区、frozen retriever、inherited topic router 和深检索预算均不训练或改动。
- 正式深检索为 `Bc=3` 个客户端、每客户端 local depth 10、传输 5 documents，总传输上限 15 documents，global pool 10。
- R3 的静态候选由继承的 P0 single-centroid profile 提供，probe 只在 `L=5` 或 `L=8` 个候选客户端上执行。
- Probe 只使用 query 与本地检索得到的标量、rank 和匹配统计；不返回正文、完整 passage、完整 embedding、answer、gold support、gold client 或 reader 信息。
- 2Wiki/MuSiQue 的 R3 Probe-Dev 为原 Router-Dev 的未使用区段 `[200:300]`，与 R2-A smoke `[0:100]` 和 R2-A.6 Recovery-Dev `[100:200]` 零重叠。
- 每个数据集完成两次 replay；除 wall-clock 计时字段外，feature transcript、upper bound、P0--P5 selection、逐 query outcome 和汇总指标均语义精确一致。
- reader 与 final test 在全部阶段保持封存。

R3 的 all-client local retrieval 仅用于离线物化 probe opportunity 和评测上限；部署合同中，server 先用 P0 确定 `L` 个候选，再仅向这些候选发 probe，最后向其中 3 个客户端请求正式 documents。不能把离线 all-client 物化的计算成本写成部署时的 probe 通信成本。

## 3. 已完成阶段与瓶颈定位

| 阶段 | 数据与设置 | 关键结果 | 结论 |
| --- | --- | --- | --- |
| M0/M1 | HotpotQA N=300, frozen `Bc=3`, depth 10, budget 15 | local complete@5/10 = 0.440/0.487；A0 raw merged@10 = 0.300；rank-percentile = 0.423（+12.3pp，37 rescue/0 harm） | 本地深度有机会；跨客户端原始分数 merge 失配真实存在；rank percentile 是稳定 label-free 修复。 |
| M0-Confirm | 2Wiki/MuSiQue N=300 | 2Wiki percentile-raw +4pp；MuSiQue +0.3pp | merge calibration 不跨数据集稳定泛化，reader 禁止。 |
| U1 | 2Wiki/MuSiQue N=300, shared BM25 top-100/local-ranker contract | inherited route coverage 0.400/0.423；oracle route 0.977/0.983；RoutingGain +20.3pp/+23.0pp | 上游 routing 是主瓶颈；该结论限定在共享 local-retrieval 候选合同。 |
| R2-A / R2-A.5 | 静态 resource card 与 Top8-to-Top3 compression | P0 Top8 complete client-set recall = 0.890/0.800；Top3 压缩后为 0.540/0.460；无训练 redundancy/prototype/lexical selector 未恢复 | 候选集合既有 absence，也有 Top8 到 Top3 的压缩损失；静态 set selector 无效。 |
| R2-A.6 REMP | 代表性证据记忆 | 2Wiki complete@5 0.810 -> 0.830；MuSiQue 0.670 -> 0.680，且 gcr -0.008；profile 约 99 KB/client 对比 P0 约 3 KB/client | 表征扩容不值得，正式停止 MARS-Route 静态 memory 路线。 |
| R3 Probe-Dev | 2Wiki/MuSiQue 各 N=100，query-time no-body probe | 见下表 | query-time probe 是第一个同时满足跨数据集 quality gate 的方向。 |

## 4. R3 方法：ProbeRoute-FedRAG

R3 不再要求静态 profile 直接在 20 个客户端中精确挑出 3 个。它把选择拆为两阶段：

1. P0 静态 single-centroid profile 先生成小候选集合 `L in {5,8}`；
2. 每个候选客户端对 query 执行浅层本地 dense/sparse 检索，仅返回可计算的机会信号；server 依据这些信号从候选中选深检索的固定 3 clients。

冻结的 probe feature family 包含 dense top-1/top-3/margin/std/entropy、BM25 top-1/top-3/margin、dense--sparse agreement、query-token/entity/title match 及 top-3 diversity。开发中比较的无训练规则为：

- `P0`：静态 P0 Top-3；
- `P1`：probe dense top-1；
- `P2`：probe dense top-3 mean；
- `P3`：BM25 anchor 在本地 dense 排名中的 percentile；
- `P4`：dense/BM25 client-rank RRF；
- `P5`：static score + normalized probe score，预注册权重 0.25、0.50、0.75。

`O2` 使用 support-derived client label 的五折 logistic cross-validation，仅为离线诊断上限，绝不作为已部署路由器。

## 5. R3 Probe-Dev 结果

### 5.1 Probe 可分性与候选上限

| Dataset | Static score AUPRC | Best probe feature | Best probe AUPRC | O0 static Top-3 | O1 oracle subset within P0 Top-8 | O2 CV diagnostic Top-3 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2Wiki | 0.406 | dense top-1 score | 0.764 | 0.500 | 0.810 | 0.490 |
| MuSiQue | 0.460 | dense top-1 score | 0.754 | 0.450 | 0.860 | 0.610 |

最重要的发现是 dense top-1 score 在两个数据集均显著优于静态 score，且 effect size 很大：2Wiki Cohen's d = 1.65，MuSiQue d = 1.75。O1 同时说明 P0 Top-8 中仍有大量可恢复空间。O2 并不总是优于最佳 label-free 规则，特别是 2Wiki 的 O2=0.490 低于 P0=0.500；因此它只能被理解为有限样本、固定 feature 下的诊断性上限，而非“训练必然带来更高收益”的证据。

### 5.2 最优无训练规则的证据传导

| Dataset | P0 coverage@3 | Best rule | Best coverage@3 | Delta (95% CI) | P0 local@10 | Best local@10 | Transmitted@15 | Raw merged@10 | Percentile merged@10 | Rescue/Harm |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 0.500 | P5, alpha=0.25 | 0.590 | +0.090 [+0.020, +0.160] | 0.170 | 0.210 | 0.210 | 0.210 | 0.210 | 12/3 |
| MuSiQue | 0.450 | P1, dense top-1 | 0.700 | +0.250 [+0.160, +0.340] | 0.180 | 0.350 | 0.310 | 0.290 | 0.270 | 26/1 |

这满足 R3 的 quality signal gate：两数据集 coverage 改善至少 5pp、local complete 改善至少 3pp，且 rescue 大于 harm。更关键的是，coverage 不是孤立指标：在 MuSiQue，+25pp routing coverage 经本地、15-document transmission 与 global merge 后，仍保留明显完整证据收益；在 2Wiki，传导较小但为正。

需要保留的限制是：最佳规则在两个数据集并不相同（2Wiki 为 P5，MuSiQue 为 P1）。这支持“probe 信号有效”，但尚不支持“单一无训练规则已经跨域定型”。下一阶段的轻量 ranker 必须在冻结训练/holdout 协议下检验是否能统一这两个 operating point。

## 6. Compact Wire-Payload Audit

首次 R3 汇总把完整 JSON debug transcript 计作通信，得到约 7.6 KB/query，接近 15-document payload。这是不正确的 wire accounting：它重复传递 schema key 与 title/entity 调试文本，而这些内容既不被 P0--P5 使用，也不应出现在部署 payload。

压缩审计保持全部 feature、规则和结果冻结，只将已有 18 个 scalar features 以 little-endian float32 固定 schema 编码：

- header：16 bytes/query；
- 每 probe client：18 x 4 = 72 bytes；
- `L=8`：16 + 8 x 72 = **592 bytes/query**；
- P0 静态基线不 probe，因此新增 probe bytes 为 0；
- server 已知 P0 candidate ID 与 static score，因此无需由 client 回传；
- 正式 payload 不含 title/entity string、正文、passage ID、embedding、gold 或 reader 字段。

| Dataset | Selected frozen rule | Compact wire bytes | Verbose debug bytes | 15-document bytes | Ratio | Float32 choice round-trip | Frozen retrieval recomputed? |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2Wiki | P5 alpha=0.25 | 592 | 7,656 | 7,474 | 7.92% | exact for all P0--P5, L=5/8 | No |
| MuSiQue | P1 dense top-1 | 592 | 7,695 | 8,818 | 6.71% | exact for all P0--P5, L=5/8 | No |

因此，`compact_probe_communication_contract_confirmed`。这项结论仅说明真实通信表示可以保持轻量且忠实复现当前规则；它不额外创造质量指标，也不替代 future holdout。

## 7. 当前决策

### 可以确认

1. 静态 profile 是当前 query-specific client selection 的主要限制，单纯加大静态 memory 并不能经济地修复它。
2. Query-time local probe 含有超出 static profile 的强相关性信号，并在两个异构多跳数据集上产生正向 complete-support 改善。
3. 该改善能在固定 `Bc=3`、15-document 预算下至少部分传导到 local、transmitted 与 merged evidence。
4. 所需 probe 可以以 592-byte fixed scalar payload 实现，低于 matched document payload 的 10%，且不传正文。

### 不能确认

1. 尚未确认统一的跨数据集无训练规则；目前的最优 rule 数据集相关。
2. 尚未确认监督 probe ranker 比简单规则更好；O2 的 2Wiki 诊断结果提示这种增益并非必然。
3. 尚未完成 fresh holdout N>=300、多 seed、Hotpot transfer、reader answer/SP/Joint 验证或端到端显著性检验。
4. 因此不能声称最终 Federated RAG QA 方法成立，也不能启动 reader。

## 8. 下一步建议与门控

当前通信审计已通过，但按照“本任务只做 wire audit”的范围，**没有启动 ranker 或 reader**。若继续，应先冻结以下轻量监督实验，而不是直接碰 reader：

1. 用已有 Router-Train / Probe-Train 训练 logistic regression、small MLP 或 LambdaMART；hard negatives 限于 P0 Top-8 中 high-static/no-support clients。
2. 固定训练目标、normalization、candidate `L`、selection rule 与 set-aware 开关；先比较 independent Top-3，只有其有效后再比较一个 relevance-minus-redundancy set rule。
3. 在新的 Probe-Holdout 上做 2Wiki 与 MuSiQue 各 N>=300、至少 3 seeds 的确认。门槛为两个数据集 coverage +8pp、local complete +5pp、merged complete +5pp，paired-bootstrap CI 下限大于 0，且无数据集退化超过 2pp。
4. 只有该 fresh-holdout 通过，才原样迁移 HotpotQA，并随后以 FLAN-T5-Large 与 UnifiedQA-T5-Large 评估完整样本。

在获得明确的下一阶段授权前，`ranker_training/` 保持未启动；所有 reader 调用继续禁止，直至 fresh-holdout 与后续 reader gate 通过。

## 9. 主要产物

- `stage_r3_probe_route/protocol/probe_preregistration.md`
- `stage_r3_probe_route/probe_features/{dataset}/feature_discrimination.csv`
- `stage_r3_probe_route/probe_oracle/{dataset}/probe_upper_bound.csv`
- `stage_r3_probe_route/label_free_baselines/{dataset}/main_results.csv`
- `stage_r3_probe_route/communication/{dataset}/compact_probe_wire_payload_audit.csv`
- `stage_r3_probe_route/communication/quality_cost_pareto.csv`
- `stage_r3_probe_route/reports/r3_probe_dev_final_report.md`
- `stage_r3_probe_route/reports/compact_probe_wire_payload_audit.md`
