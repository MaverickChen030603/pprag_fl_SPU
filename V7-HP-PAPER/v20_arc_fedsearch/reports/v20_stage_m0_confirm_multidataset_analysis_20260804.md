# V20 Stage M0-Confirm 实验分析报告

**日期：** 2026-08-04  
**状态：** `routing_residual_reconfirmed`  
**结论：** rank-percentile 合并在 HotpotQA 上有清晰、可重复的正信号，在 2WikiMultiHopQA 上有条件性改善，但未达到预注册的跨数据集门槛；MuSiQue 上未复现有效增益。因此，本阶段不启动 reader。

## 1. 实验目的

V20 不训练新的检索器、路由器、分配器或校准器，而是检验一个更基础的问题：当每条多跳查询已经由冻结的联邦路由器选定三个客户端时，跨客户端的**分数尺度不一致**是否会使本应已在本地候选中出现的两跳证据，在服务器端 top-10 合并中被错误丢弃？

实验严格区分四段损失：

1. 本地深度机会：选中客户端的 local top-5 扩展至 local top-10 后，双支持文档能否同时出现；
2. 15-document 传输损失：local top-10 的完整证据是否在固定通信预算下仍被传输；
3. 原始跨客户端合并损失：传输后的证据是否被 raw score top-10 丢弃；
4. 无标签校准收益：仅用本地 rank percentile 替代不可比较的原始分数后，能否恢复这些证据。

这是一项 retrieval-only 验证；它不回答下游答案是否改善，也不以 reader、answer 或 support 标签构造任何检索、路由、分配或校准特征。

## 2. 冻结实验合同与审计

| 项目 | 固定设置 |
|---|---|
| 数据 | HotpotQA、2WikiMultiHopQA、MuSiQue 的 development-only 冻结 N=300 切片（行 101--400） |
| 路由 | V17 inherited origin-plus-topic-centroid router |
| 客户端预算 | `Bc=3`，每条查询固定 `selected_clients` |
| 本地候选 | 每个物理客户端 local depth=10；all-client 仅用于候选物化 |
| 通信预算 | 15 文档 |
| 服务器上下文池 | global top-10 |
| 主分配 | A0：每个选中客户端各传 5 篇（5/5/5） |
| 主合并 | M1：rank-percentile；不使用 gold、answer、reader 或训练参数 |
| 对照 | M0 raw score；A1 source-confidence proportional；A2/A3/A4 简单规则；A5 仅供 oracle 分析 |
| 可复现性 | 每数据集完整独立运行两次；矩阵与逐 query 输出均要求字节一致 |
| Reader | 禁用，未运行 |

每个数据集两次运行的 `allocation_merge_matrix.csv` 与 `per_query_allocation_merge.csv` 均 byte-identical。正式统计始终使用冻结 Bc=3；“all-client”不会进入正式 route 指标，也不改变客户端集合。

## 3. 主结果

主指标为 `complete-support@10`：一条查询的所有 gold supporting documents 是否同时位于最终 top-10。数值是 N=300 的比例。

| 数据集 | client coverage@3 | Local@5 | Local@10 | A0 transmitted@15 | A0 raw@10 | A0 percentile@10 | Delta | Rescue / Harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.707 | 0.440 | 0.487 | 0.440 | 0.300 | 0.423 | **+0.123** | 37 / 0 |
| 2WikiMultiHopQA | 0.400 | 0.157 | 0.170 | 0.157 | 0.117 | 0.157 | +0.040 | 12 / 0 |
| MuSiQue | 0.423 | 0.137 | 0.190 | 0.137 | 0.120 | 0.123 | +0.003 | 4 / 3 |

辅助 support recall@10 的变化为：HotpotQA `0.567 -> 0.630`（+0.063），2Wiki `0.378 -> 0.401`（+0.023），MuSiQue `0.411 -> 0.410`（约 -0.0003）。因此，MuSiQue 的 complete-support 微小变化不能被解读为稳定的 evidence 增益。

### HotpotQA：合并校准是清晰、但数据集特异的正信号

- local depth 从 5 增至 10，使完整证据从 `0.440` 到 `0.487`，增加 14 条查询（+4.67pp）。
- A0 的 15-document transmission 保留 `0.440`，说明其中 14 条深层机会在 5/5/5 预算下仍未传出。
- raw merge 将完整证据从 `0.440` 压至 `0.300`，产生 42 条传输了但被服务器排序丢失的查询。
- rank-percentile 恢复其中 37 条且未伤害任何 raw 成功查询，最终至 `0.423`（相对 raw +12.33pp）。
- 剩余 19 条 local@10 成功但 percentile 失败的案例中，14 条属于 allocation loss，5 条属于 calibrated merge residual。

这确认了 HotpotQA 上的主要机制：在既有 Bc=3 route 已覆盖两跳证据的前提下，客户端内部归一化分数不能直接跨客户端比较；rank-based calibration 能有效修复该后处理错误。

### 2WikiMultiHopQA：条件性正信号被低路由覆盖限制

- raw 到 percentile 为 `0.117 -> 0.157`（+4.00pp），有 12 条 rescue、0 条 harm；方向与 HotpotQA 一致。
- 但未达到预注册的 `>=5pp` 门槛；更重要的是，`client coverage@3=0.400`，而 local@10 仅 `0.170`。
- A0 percentile 恰好达到传输上限 `0.157`：对已经传出的证据，校准几乎完全修复了 raw merge 损失；但绝大多数查询的双跳证据在冻结 Bc=3 路由下不可得。

因此，2Wiki 的结果支持“merge calibration 在可用证据上是有用的”，但不能支持“它已经构成跨域、匹配预算下的稳定方法收益”。

### MuSiQue：深度机会存在，合并校准未能转化为收益

- local@5 到 local@10 为 `0.137 -> 0.190`，新增 16 条机会，说明局部深度本身不是无效的。
- 固定 A0 transmission 只保留 `0.137`，局部机会到 15-document 预算的损失为 5.33pp，超过 raw merge 损失。
- M1 只从 `0.120` 至 `0.123`，4 条 rescue 同时有 3 条 harm，support recall 还略降。
- A1 在 M1 下为 `0.127`，略高于 A0 M1 的 `0.123`，但幅度仅 1 条查询，且未在第二个数据集复现；不构成 allocation rule 的行动依据。

MuSiQue 说明：将本地 rank 映射为 percentile 并非普遍适用的跨客户端分数校准。该数据集的主要瓶颈更靠前，位于路由覆盖和固定 15-document allocation，而不是 raw merge 本身。

## 4. 简单 allocation 与 oracle 分析

在 HotpotQA，A1 的 transmission retention 从 `0.440` 提至 `0.453`，raw merge 从 `0.300` 提至 `0.397`；但是 A1 percentile=`0.420`，略低于 A0 percentile=`0.423`。A2/A3/A4 也均未超过 A0 的主配置。2Wiki 同样是 A0 percentile=`0.157` 略优于 A1=`0.153`。MuSiQue A1 的微弱优势只对应一条查询。

故没有证据表明“简单 source-confidence allocation + percentile”是跨数据集的主配置。A5 oracle allocation 仅用于诊断，不能部署；它也不能跨越 Bc=3 路由未覆盖支持客户端这一上游约束。

## 5. 结论边界

**可确认的结论：**

1. HotpotQA 上，冻结联邦 route 下的 raw cross-client score merge 确实会丢失已传输的多跳证据；无标签 rank-percentile 可稳定恢复该损失。
2. 该信号在 2Wiki 上方向一致，且 rescue 大于 harm，但不足预注册幅度。
3. local depth=10 在三个数据集均创造额外的局部机会，但机会能否成为最终完整支持，取决于 Bc=3 覆盖、15-document 分配与合并的共同作用。

**不能声称的结论：**

1. 不能称 rank-percentile 是跨数据集已泛化的联邦检索方法；MuSiQue 不支持该表述。
2. 不能声称下游 QA 或 reader 指标改善；reader 尚未、也不应在当前 gate 下运行。
3. 不能将 A5 oracle 的结果视作可部署性能，也不能用 all-client materialization 代替 Bc=3 正式路由。

## 6. 决策与下一步

预注册要求至少两个数据集同时满足：`percentile - raw >= 0.05`、rescue > harm、local@10 > local@5、`Bc=3`、15-document budget 及可重复性。当前只有 HotpotQA 满足，最终状态为：

> **`routing_residual_reconfirmed`**

reader 继续禁止。下一步应是一个独立的 retrieval-only 路由审计，而非训练 learned calibrator 或立即上 reader：

1. 对 2Wiki/MuSiQue 做冻结 Bc=3 routing gap / oracle@Bc=3 分解，量化缺失支持客户端的比例；
2. 分析支持文档跨 client 的 shard 分布与 origin-plus-centroid route 的覆盖失配；
3. 只有在不改变 partition、并在独立冻结开发协议上证明 recall-first 路由能提高 coverage@3 后，才决定是否开发新的 router；
4. 继续保留 Hotpot percentile 分支作为“后处理校准机制”的条件性证据，而不是主方法的跨域结论。

## 7. 产物与代码版本

- 主表：`multidataset_depth_calibration/main_results.csv`
- 逐 query 结果：`multidataset_depth_calibration/per_query_results.jsonl`
- 可复现性哈希：`multidataset_depth_calibration/reproducibility.json`
- Hotpot 归因：`hotpot_error_analysis/`
- gate 决策：`reports/reader_start_decision.json` 与 `reports/next_module_decision.md`
- 实现与冻结合同已推送至 GitHub commit `d3c0511` (`feat(v20): add frozen multidataset merge confirmation`)。
