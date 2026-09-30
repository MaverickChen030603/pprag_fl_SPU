# V20-B3F-M 独立直接竞争基线确认实验：完整审计报告

## 审计结论

本实验在一个未被 M2、RAGRoute、R5、C1、C2 或 B3F 使用过的独立新鲜查询集上，直接比较 M2 ProbeRoute Logistic 与 `RAGRoute-Fixed3 (protocol-adapted)`。两者共享同一冻结 Top-8 候选池、选择 3 个客户端、每客户端 5 篇文档、本地 depth 10、15 篇上传、raw-score Top-10 merge、Top-5 Reader context，以及相同的 FLAN-T5-Large/UnifiedQA-T5-Large 推理设置。

结论为 `direct_competitor_confirmation_passed`：M2 在三个数据集上均提高 Context Complete@5；在 HotpotQA 与 2Wiki 上的双 Reader Joint F1 均显著优于 RAGRoute-Fixed3，在 MuSiQue 上 FLAN 的提升显著、UnifiedQA 方向为正但未达显著。该结论仅限于本项目的联邦多跳 QA 设置和协议适配版 RAGRoute，不外推为对 RAGRoute 的一般性结论。

## 1. 研究问题与预注册合同

核心问题：在完全一致的文档传输预算下，利用 18 维客户端本地 query-time probe 与静态分数的 M2，能否比利用 query embedding、source centroid 和 source ID 的 RAGRoute-Fixed3 更好地选择带有完整证据的客户端，并提升端到端 RAG QA？

主方法：B0 Static Top-3、RAGRoute-Fixed3、M2 ProbeRoute Logistic、B4a All-Candidate Top-8 高成本参考；RAGRoute-Threshold 仅作为动态成本--质量参考，不进入同预算主显著性比较。

冻结合同：P=8，B=3，local depth=10，5 docs/client，15 docs/query，raw dense-score Top-10 merge，Top-5 Reader context。M2 未重训；RAGRoute 使用 B3F 已冻结的 `1556→256→128→1`、三种子 mean-sigmoid 集成；Reader、prompt、截断和生成参数均未修改。

## 2. Freshness 与训练公平性

每个数据集选取 500 条公开训练池查询，共 1,500 条。选择使用固定 salted SHA-256 排序，并排除历史训练、历史开发/方法选择/已揭示集合、M2 probe train、RAGRoute train、C1、C2 和 B3F。对 query ID 与 normalized question hash 双重审计，三数据集所有交集均为 0。

审计文件：

- `splits/b3fm_fresh_split_manifest.json`
- `splits/b3fm_zero_overlap_audit.json`
- `protocol/training_fairness_audit.md`
- `protocol/b3fm_frozen_method_contract.json`

M2 与 RAGRoute 均使用同一历史 5,000-query、P0 Top-8 训练宇宙及“客户端含至少一个 canonical support document”的 source-level 监督定义；B3F-M 查询未进入任一训练集。

## 3. 盲态流程与完整性

执行顺序为：新鲜性审计与预注册、question-only 输入、三数据集 Probe packet、五方法盲态路由和上下文物化、FLAN/UnifiedQA 无评分预测及 support prediction、SHA-256 gate、人工 `UNSEAL B3FM LABELS`、一次性评分。

预解封 gate 通过：5 个方法 × 3 数据集 × 500 查询 = 7,500 个唯一上下文键；两个 Reader 各有 7,500 条预测且均含 `predicted_support`；所有盲态记录均为 `labels_loaded=false`、`metrics_computed=false`。

关键哈希在 `checksums/b3fm_pre_unseal_manifest.json`；解封记录在 `protocol/b3fm_label_unseal_record.json`；评分清单在 `statistics/evaluation_manifest.json`。

## 4. 合同修订与异常透明度

首次盲态 Reader 运行发现 OriginalThreshold 合法选择 0 客户端时，旧 C1/C2 support extractor 对空上下文未定义。此时未解封、未评分；各 Reader 仅有 24 条 partial 预测，均被隔离到 `quarantined/pre_amendment/`。

修订 `amendment_01_empty_context_support_fallback.md` 明确：当 `reader_context_docs=[]` 时，确定性写出 `predicted_support=[]`。该修订不改变路由模型、候选 ID、阈值、预算、检索、merge、Reader prompt 或生成；随后从路由物化开始完整重跑全部盲态阶段。该事件及原始停止理由记录于 `protocol/integrity_exception.md`。

## 5. 主结果

| Dataset | Method | Context Complete@5 | FLAN Joint F1 | UnifiedQA Joint F1 | Docs/query |
| --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | B0 | 0.474 | 0.2627 | 0.2246 | 15.00 |
| HotpotQA | RAGRoute-Fixed3 | 0.492 | 0.2703 | 0.2298 | 15.00 |
| HotpotQA | M2 | 0.546 | 0.2901 | 0.2531 | 15.00 |
| HotpotQA | B4a | 0.544 | 0.2903 | 0.2529 | 40.00 |
| HotpotQA | Threshold | 0.338 | 0.1990 | 0.1714 | 6.36 |
| 2Wiki | B0 | 0.178 | 0.1500 | 0.1125 | 15.00 |
| 2Wiki | RAGRoute-Fixed3 | 0.198 | 0.1655 | 0.1346 | 15.00 |
| 2Wiki | M2 | 0.242 | 0.2014 | 0.1637 | 15.00 |
| 2Wiki | B4a | 0.244 | 0.2042 | 0.1607 | 40.00 |
| MuSiQue | RAGRoute-Fixed3 | 0.142 | 0.0874 | 0.0719 | 15.00 |
| MuSiQue | M2 | 0.176 | 0.0998 | 0.0816 | 15.00 |
| MuSiQue | B4a | 0.172 | 0.0955 | 0.0813 | 40.00 |
| MuSiQue | Threshold | 0.112 | 0.0745 | 0.0601 | 7.90 |

完整逐查询结果在 `statistics/per_query_results.csv`，完整五方法表在 `statistics/main_results.csv`。

## 6. M2 相对 RAGRoute-Fixed3 的主统计

| Dataset | Metric | Delta | 95% bootstrap CI | Holm-adjusted p |
| --- | --- | ---: | --- | ---: |
| HotpotQA | Context Complete@5 | +0.054 | [+0.034, +0.076] | 0.0018 |
| 2Wiki | Context Complete@5 | +0.044 | [+0.024, +0.066] | 0.0018 |
| MuSiQue | Context Complete@5 | +0.034 | [+0.018, +0.052] | 0.0018 |
| HotpotQA FLAN | Joint F1 | +0.0197 | [+0.0085, +0.0314] | 0.0018 |
| HotpotQA UnifiedQA | Joint F1 | +0.0233 | [+0.0129, +0.0346] | 0.0018 |
| 2Wiki FLAN | Joint F1 | +0.0359 | [+0.0204, +0.0513] | 0.0018 |
| 2Wiki UnifiedQA | Joint F1 | +0.0292 | [+0.0136, +0.0443] | 0.0018 |
| MuSiQue FLAN | Joint F1 | +0.0125 | [+0.0025, +0.0224] | 0.0256 |
| MuSiQue UnifiedQA | Joint F1 | +0.0096 | [-0.0005, +0.0205] | 0.0656 |

统计使用查询级配对差异、5,000 次 bootstrap、5,000 次 randomization，并对 3 个 context tests 加 6 个 Reader-specific Joint tests 的 9 项主检验实施 Holm 校正。详见 `statistics/m2_vs_ragroute_bootstrap.csv` 与 `statistics/holm_primary_tests.csv`。

## 7. 成本解释

M2 和 RAGRoute-Fixed3 同为 3 个深检索客户端和 15 篇文本传输，因此主质量结论不由文档 payload 增加造成。二者不同在预路由信息：RAGRoute 在服务端使用 query embedding、预存 source centroids/source IDs 与 MLP；M2 需向 8 个候选客户端发起 shallow probe，返回 `8×18×float32=576 bytes/query` 加协议字段，再以静态分数和 probe 特征进行 logistic 排序。

B4a 传输 40 篇文档，但 M2 在 HotpotQA/2Wiki 上接近其 Context Complete 和 Joint F1，在 MuSiQue 上也与其相当或略高。Threshold 的约 6.36--7.90 篇低 payload 伴随明显证据与 QA 损失，应定位为 low-cost/fragile operating point，而非 matched-budget 主竞争者。

## 8. 论文可用表述与限制

可表述：在本项目的联邦多跳 QA 协议下，M2 在严格匹配的文档返回预算中显著提升完整证据恢复，并在多数 Reader-dataset cells 上带来更高 Joint F1；其代价是 query-time client-local probe metadata 与浅层客户端计算。

不可表述：M2 一般性地优于所有 RAGRoute 实现；B4a 是 oracle/global upper bound；Threshold 的低文档数等价于同质量通信节省。

限制：本次修订在盲态阶段补足了空上下文 support schema，并已完整隔离/重跑；该修订应在论文附录与审稿材料中透明披露。成本审计中的实际 UTF-8 document bytes、协议头和 simulation latency 应在后续 `cost/communication_compute_audit.csv` 中补齐，不能以文档数替代字节级成本。

## 9. 审计文件索引

- Pre-registration: `protocol/b3fm_preregistration.md`
- Frozen contract: `protocol/b3fm_frozen_method_contract.json`
- Amendment: `protocol/amendment_01_empty_context_support_fallback.md`
- Pre-unseal manifest: `checksums/b3fm_pre_unseal_manifest.json`
- Unseal record: `protocol/b3fm_label_unseal_record.json`
- Results: `statistics/main_results.csv`, `statistics/per_query_results.csv`
- Primary statistics: `statistics/m2_vs_ragroute_bootstrap.csv`, `statistics/holm_primary_tests.csv`
