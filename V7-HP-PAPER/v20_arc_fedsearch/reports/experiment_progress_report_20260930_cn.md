# ProbeRoute 实验进展报告

鉴定日期：2026-09-30

## 1. 当前总体状态

ProbeRoute 的主确认链已完成：C1/C2 承担宽基线的新鲜确认，B3F 完成 protocol-adapted RAGRoute 的阈值规则复现，B3F-M 在一个新的独立 fresh split 上完成 M2 ProbeRoute 与 RAGRoute-Fixed3 的严格同预算直接竞争比较。P0 已完成同一 B3F-M 查询集上的应用层通信 payload、离线存储和已记录 probe simulation latency 审计。

当前可支持的核心结论是：在本项目的联邦多跳 QA 协议中，M2 的质量提升并非来自更多正文文档传输；相对于同样传输 15 篇文档的 RAGRoute-Fixed3，M2 在独立 1,500-query 确认集上提高完整证据恢复，并在多数 Reader-dataset cell 上提高 Joint F1。其代价是 8-client query-time shallow probe 和额外 probe metadata 通信。

## 2. 已完成实验

### 2.1 C1/C2：宽基线新鲜确认

C1 与 C2 使用独立 fresh split，完成 B0、随机路由、旧 RAGRoute-style development baseline、B6、M2 和 B4a 的宽度比较，并采用双 Reader 与解封后统计。其角色是证明 M2 相对传统静态/开发基线的有效性，并展示其接近 B4a 高成本参考的能力。

这些结果不与 B3F-M 混称为同一次预注册实验：C1/C2 是 broad baseline confirmation，B3F-M 是 additional independently frozen direct-competitor confirmation。

### 2.2 B3F：RAGRoute 规则与成本--质量 operating point

B3F 在 3 数据集 × 500 fresh queries 上复现 protocol-adapted RAGRoute 的两种固定规则：

- `RAGRoute-Fixed3`：三种子 mean-sigmoid 排序后选 Top-3。
- `RAGRoute-OriginalThreshold`：mean sigmoid 大于 0.5 的客户端，无回退。

Threshold 的平均文档传输为 HotpotQA `6.21`、2Wiki `8.30`、MuSiQue `7.87`，显著低于 Fixed3 的 15 篇；但三个数据集的 Reader Context Complete@5 均下降，HotpotQA/MuSiQue 的 Answer F1 也总体下降。因此 Threshold 应定位为 low-cost/fragile operating point，而不是 matched-budget 主竞争者。

审计报告：`stage_b3f_ragroute_faithful/reports/B3F_RAGRoute_protocol_faithful_audit_report_cn.md`。

### 2.3 B3F-M：M2 vs RAGRoute-Fixed3 的直接确认

B3F-M 新建了独立 1,500-query fresh split：每数据集 500 条，和 M2 train、RAGRoute train、历史训练/开发/方法选择/已揭示集合、C1、C2、B3F 进行 query ID 与 normalized question hash 双重零重叠审计。

五种冻结方法：B0 Static Top-3、RAGRoute-Fixed3、M2 Logistic ProbeRoute、B4a All-Candidate Top-8、RAGRoute-Threshold。主同预算比较只针对 M2 与 RAGRoute-Fixed3：P=8、B=3、local depth=10、5 docs/client、15 docs/query、raw-score Top-10 merge、Top-5 Reader context、相同 FLAN/UnifiedQA。

盲态产物为 7,500 个唯一 method-query contexts 和每个 Reader 7,500 条包含 answer/support prediction 的输出；均通过 SHA-256 pre-unseal gate。解封后产生 15,000 条逐查询统计记录。

#### B3F-M 主结果

| Dataset | M2 - RAGRoute Context Complete@5 | FLAN Joint F1 Delta | UnifiedQA Joint F1 Delta |
| --- | ---: | ---: | ---: |
| HotpotQA | +0.054, Holm p=0.0018 | +0.0197, Holm p=0.0018 | +0.0233, Holm p=0.0018 |
| 2WikiMultiHopQA | +0.044, Holm p=0.0018 | +0.0359, Holm p=0.0018 | +0.0292, Holm p=0.0018 |
| MuSiQue | +0.034, Holm p=0.0018 | +0.0125, Holm p=0.0256 | +0.0096, Holm p=0.0656 |

所有 Context Complete@5 的 95% bootstrap CI 均在 0 以上。HotpotQA 与 2Wiki 的双 Reader Joint F1 均显著支持 M2；MuSiQue 的 FLAN 支持 M2，UnifiedQA 方向为正但区间跨零。因此整体满足预注册的 `direct_competitor_confirmation_passed` 标准。

M2 在 HotpotQA/2Wiki 上接近 40-document B4a 参考，同时仅传输 15 篇文档；Threshold 虽只传约 6.36--7.90 篇，但质量明显较弱。

完整报告：`stage_b3fm_direct_competitor_confirmation/reports/B3FM_direct_competitor_audit_report_cn.md`。

## 3. 完整性异常与处理

B3F-M 的首次盲态 Reader 运行发现：RAGRoute-Threshold 在合法零客户端选择时给出空 Reader context，而复用的 C1/C2 support extractor 未定义空输入行为。系统在每个 Reader 仅写入 24 条 partial 输出时停止；未解封、未评分。

随后采用透明修订：明确 `reader_context_docs=[] -> predicted_support=[]`，不改动任何路由模型、阈值、候选池、Retriever、预算、merge、Reader prompt 或 generation。此前 contexts/predictions 被隔离，五方法路由、双 Reader 与 support prediction 从头重跑，并重新完成 SHA-256 gate。修订和异常记录分别保存在：

- `protocol/integrity_exception.md`
- `protocol/amendment_01_empty_context_support_fallback.md`

这一修订必须在论文附录和导师审计材料中如实披露。

## 4. P0 系统成本审计

P0 直接复用 B3F-M 的 1,500 queries、冻结候选池、选择结果和 transmitted docs。它未重跑或改动方法。

已完成：

- 真实文档 payload：`UTF-8(title + "\\n" + text)`。
- query/request payload、M2 probe response `8×18×4=576 bytes/query`、统一 JSON diagnostic header、application-level total bytes。
- RAGRoute centroids/MLP 与 M2 logistic 等离线持久化成本。
- 已记录的 8-candidate probe materialization warm-cache simulation latency。

通信结果显示，M2 与 RAGRoute-Fixed3 都传 15 篇正文文档，但 M2 的 application payload 约高 25%--32%，原因是向 8 客户端下发 query 与返回 576-byte probe metadata。该成本应与质量提升一起报告，不能隐藏。

当前无法合法报告的内容：真实 wire-level bytes、统一的 deep-retrieval critical-path latency、Reader critical-path latency。现有系统是单机 filesystem/shared-memory simulation，缺少真实 RPC 或对所有方法公平的统一微基准；这些项已明确标为 `unavailable`，未以估算值替代。

P0 产物：

- `cost/per_query_communication_compute_audit.csv`
- `cost/communication_compute_audit.csv`
- `cost/recorded_latency_audit.csv`
- `cost/offline_storage_audit.csv`
- `cost/environment_manifest.json`

## 5. 论文当前可写主张

推荐主张：在所定义的联邦多跳 QA 设置中，ProbeRoute 以 query-conditioned local probe evidence 改善 evidence-bearing client selection；在相同 15-document return budget 下，M2 在独立 fresh confirmation 上显著改善 Context Complete@5，并在多数双 Reader Joint F1 cell 中优于 protocol-adapted RAGRoute-Fixed3。

必须同步说明：M2 并非“免费”提升。它需要 8-client shallow probe、逻辑 query broadcast、576-byte probe response 及额外协议字段；RAGRoute 的 centroids/MLP 主要是 server-side/offline 成本。论文应将质量--通信--客户端计算 trade-off 透明地呈现。

不可使用的表述：ProbeRoute 一般性地优于 RAGRoute；B4a 是 oracle/global upper bound；Threshold 在同质量下节省通信；P0 的 simulation latency 是真实网络 latency。

## 6. 下一步建议

1. 以 B3F-M 作为论文中 M2 vs RAGRoute-Fixed3 的直接竞争主表，以 C1/C2 补充宽基线。
2. 用 P0 的 application payload 表和 offline-storage 表补齐 cost-quality trade-off 表；将未测到的 wire/deep latency 清楚列为限制。
3. 若投稿前时间允许，新增一个所有方法统一的 warm-cache deep-retrieval microbenchmark，记录每 query 3--5 次 median、P50/P95、serial 与 ideal-parallel critical path；否则不要做延迟主张。
4. 将完整性异常修订、预解封 manifest、人工解封记录和代码 commit hash 放入补充材料，增强可复现性与审计透明度。
