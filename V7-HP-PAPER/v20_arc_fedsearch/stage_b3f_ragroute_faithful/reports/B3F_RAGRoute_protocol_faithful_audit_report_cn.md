# V20-B3F：Protocol-Faithful RAGRoute 基线复现实验审计报告

## 1. 审计摘要

本报告记录 V20-B3F-Fresh 的完整执行链：先冻结独立测试查询和所有模型/预算规则，再在不读取测试标签、答案、支持证据或指标的条件下完成本地检索探测、RAGRoute 路由、上下文物化和双 Reader 预测；最后在收到明确授权 `UNSEAL B3F LABELS` 后进行一次性评估。

实验比较的是两种由同一三随机种子 RAGRoute 集成产生的路由规则：

1. `RAGRoute-Fixed3`：按平均 sigmoid 概率降序、客户端 ID 打破并列，固定选择 3 个客户端。
2. `RAGRoute-OriginalThreshold`：保留冻结 P0 Top-8 的原始顺序，选择平均 sigmoid 概率严格大于 0.5 的客户端；**不设置任何回退**。

主要结论是：`OriginalThreshold` 将平均传输文档数从固定的 15 篇降低至 6.21--8.30 篇，但在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 上均降低 Reader Top-5 的完整证据恢复率。其 Answer F1 仅在 2Wiki+FLAN 上出现很小的正向点估计，95% bootstrap 置信区间跨越 0；其他数据集/Reader 组合均不支持其 QA 效果优于 `Fixed3`。

## 2. 研究问题与比较范围

本实验回答如下问题：在既有联邦检索协议、同一候选客户端集、同一 RAGRoute 模型集成、同一本地深度和同一 Reader 的条件下，论文式概率阈值路由是否能以更低的通信量维持固定三客户端路由的证据恢复和问答效果？

本实验是对 RAGRoute 路由规则的协议忠实复现与外部比较，不是对当前 M2 ProbeRoute 方法的再优化实验。结论只适用于本报告中列明的独立 B3F-Fresh 样本、冻结 Retriever/Reader、预算和统计口径。

## 3. 冻结设计

### 3.1 数据与隔离

每个数据集从公开训练池中选取 500 条此前未使用查询，共 1,500 条：HotpotQA、2WikiMultiHopQA、MuSiQue 各 500 条。选择规则是以固定 SHA-256 salt 排序，并同时排除历史训练、C1、C2 的 query ID 与归一化问题文本哈希。冻结文件为：

`protocol/b3f_fresh_split_manifest.json`

该清单记录了样本 ID、公开来源行号和问题哈希，但不在盲态阶段打开答案或支持证据。零重叠审计记录在：

`protocol/b3f_fresh_zero_overlap_audit.json`

### 3.2 历史训练与模型

每个数据集使用已恢复的历史 P0 训练特征：5,000 条历史查询、每条 8 个候选客户端、共 40,000 个候选行。按查询级划分 4,500 条训练 / 500 条 RouterDev；StandardScaler 仅拟合训练查询。模型为 `1556 -> 256 -> 128 -> 1` MLP，包含 LayerNorm、ReLU 和 Dropout(0.4)，使用 BCE、类别权重、Adam、CyclicLR/StepLR，训练 150 epochs。

训练种子为 `0, 1, 2`。推理时对三个 checkpoint 的 sigmoid 概率取均值，禁止选取最优种子。训练合同和产物在：

`protocol/b3f_training_contract.json`

`models/{dataset}/seed_{0,1,2}.pt`

`models/{dataset}/scaler.npz`

### 3.3 冻结检索与通信协议

每条 B3F-Fresh 查询的候选客户端是冻结的 P0 Top-8。每个被选客户端本地 dense depth 为 10；每个客户端上传其前 5 篇文档。协调端按原始 dense score 合并、取 Top-10，并选前 5 篇构造 Reader 输入。

`Fixed3` 始终传输 `3 x 5 = 15` 篇文档；`OriginalThreshold` 可选择 0 个或多个客户端，且没有为填补预算而增加客户端的回退规则。两者共享完全相同的 BGE 编码器、P0 数据包、合并规则和 Reader 输入截断规则。

### 3.4 盲态与解封边界

盲态阶段的输入只包含 `dataset`、`query_id` 和 `question`；路由与 Reader 脚本不会导入评估代码或公开训练源中的标签字段。上下文物化后产生 3,000 行：

`3 datasets x 500 queries x 2 routes = 3,000`

随后 FLAN-T5-Large 与 UnifiedQA-T5-Large 均以 greedy decoding、最多 32 个新 token、最多 1,024 输入 token 执行答案预测。为了避免 Reader 阶段为预测 supporting facts 而打开原始行中的答案/支持注释，盲态 Reader 只写出答案预测；这在解封前保证标签防火墙完整，但意味着 B3F 后验评估不报告 `SP F1` 或 `Joint F1`。

用户于 2026-09-29 明确授权 `UNSEAL B3F LABELS`。该授权后的不可逆记录在：

`protocol/b3f_label_unseal_record.json`

## 4. 产物完整性与可复核性

| 产物 | 行数 | SHA-256 |
| --- | ---: | --- |
| `contexts/b3f_ragroute_contexts_unscored.jsonl` | 3,000 | `473c3f65cba00648c1dfa02207986f5bd3ece83723d96be0b3f79eb5623ce5fb` |
| `predictions/flan_b3f_unscored.jsonl` | 3,000 | `34823c3c005aedc51b90a059273647e694735b8d378638856c0cc989e8a68752` |
| `predictions/unifiedqa_b3f_unscored.jsonl` | 3,000 | `dabb40d950071db751a8d43208b39657a7f7efd1facbcc74d274d1851083c5a3` |

解封后评分产生 6,000 条逐查询、逐 Reader 结果：

`statistics/per_query_results.csv`

聚合结果、配对 bootstrap 比较和评分清单分别为：

`statistics/method_results.csv`

`statistics/paired_comparisons.csv`

`statistics/evaluation_manifest.json`

## 5. 指标定义

对每条查询定义完整证据为所有 gold supporting documents 均出现在相应集合中。报告：

- `client_complete_selected`：所有支持文档所属客户端均被选择。
- `local_complete_at_10`：被选客户端的本地 Top-10 并集包含完整证据。
- `transmitted_complete`：实际上传的全部文档包含完整证据。
- `merged_complete_at_10`：全局合并 Top-10 包含完整证据。
- `reader_context_complete_at_5`：Reader 输入 Top-5 包含完整证据。
- `answer_f1`：使用既有 `official_metrics` 对冻结 Reader 答案进行官方答案 F1 计算。

`OriginalThreshold - Fixed3` 的差异以查询级配对、5,000 次 percentile bootstrap 计算 95% CI。由于答案预测未生成 supporting-fact 输出，`sp_f1`、`joint_f1` 未计算，不能从本报告推断这两项指标。

## 6. 解封后主结果

### 6.1 聚合表现

| 数据集 | Reader | 路由 | 平均选客户端 | 平均传输文档 | Reader Top-5 完整率 | Answer F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | FLAN | Fixed3 | 3.000 | 15.00 | 0.422 | 0.5444 |
| HotpotQA | FLAN | OriginalThreshold | 1.242 | 6.21 | 0.328 | 0.4913 |
| HotpotQA | UnifiedQA | Fixed3 | 3.000 | 15.00 | 0.422 | 0.4860 |
| HotpotQA | UnifiedQA | OriginalThreshold | 1.242 | 6.21 | 0.328 | 0.4470 |
| 2WikiMultiHopQA | FLAN | Fixed3 | 3.000 | 15.00 | 0.200 | 0.4442 |
| 2WikiMultiHopQA | FLAN | OriginalThreshold | 1.660 | 8.30 | 0.134 | 0.4569 |
| 2WikiMultiHopQA | UnifiedQA | Fixed3 | 3.000 | 15.00 | 0.200 | 0.3891 |
| 2WikiMultiHopQA | UnifiedQA | OriginalThreshold | 1.660 | 8.30 | 0.134 | 0.3743 |
| MuSiQue | FLAN | Fixed3 | 3.000 | 15.00 | 0.146 | 0.2544 |
| MuSiQue | FLAN | OriginalThreshold | 1.574 | 7.87 | 0.106 | 0.2245 |
| MuSiQue | UnifiedQA | Fixed3 | 3.000 | 15.00 | 0.146 | 0.1908 |
| MuSiQue | UnifiedQA | OriginalThreshold | 1.574 | 7.87 | 0.106 | 0.1858 |

### 6.2 配对差异：OriginalThreshold 减 Fixed3

| 数据集 | Reader | 指标 | 差异 | 95% CI |
| --- | --- | --- | ---: | --- |
| HotpotQA | FLAN | Top-5 完整率 | -0.0940 | [-0.1200, -0.0680] |
| HotpotQA | FLAN | Answer F1 | -0.0531 | [-0.0804, -0.0261] |
| HotpotQA | UnifiedQA | Top-5 完整率 | -0.0940 | [-0.1200, -0.0700] |
| HotpotQA | UnifiedQA | Answer F1 | -0.0390 | [-0.0642, -0.0141] |
| 2WikiMultiHopQA | FLAN | Top-5 完整率 | -0.0660 | [-0.0920, -0.0440] |
| 2WikiMultiHopQA | FLAN | Answer F1 | +0.0127 | [-0.0089, +0.0344] |
| 2WikiMultiHopQA | UnifiedQA | Top-5 完整率 | -0.0660 | [-0.0900, -0.0420] |
| 2WikiMultiHopQA | UnifiedQA | Answer F1 | -0.0148 | [-0.0409, +0.0113] |
| MuSiQue | FLAN | Top-5 完整率 | -0.0400 | [-0.0600, -0.0200] |
| MuSiQue | FLAN | Answer F1 | -0.0300 | [-0.0511, -0.0093] |
| MuSiQue | UnifiedQA | Top-5 完整率 | -0.0400 | [-0.0600, -0.0200] |
| MuSiQue | UnifiedQA | Answer F1 | -0.0050 | [-0.0223, +0.0122] |

## 7. 解释与结论

`OriginalThreshold` 的主要收益是通信节省：相对固定 15 篇文档，HotpotQA、2WikiMultiHopQA、MuSiQue 分别少传约 58.6%、44.7%、47.5% 的文档。代价同样清晰：每个数据集的 Reader Top-5 完整证据恢复率均下降，且对应 bootstrap 区间均在 0 以下。

从下游 QA 看，HotpotQA 和 MuSiQue 的 FLAN 结果明确偏向 `Fixed3`；HotpotQA 的 UnifiedQA 也明确偏向 `Fixed3`。2Wiki 上 Threshold 的 FLAN 点估计略高，但其 95% CI 跨越 0，不能作为阈值路由优于固定路由的证据。综合两 Reader、三数据集与上游证据链条，本审计不支持“固定 0.5 阈值能在显著节省通信的同时保持同等端到端效果”的主张。

合理的论文表述是：论文式阈值策略形成了一个低通信但明显更脆弱的对照点；在需要稳定多跳证据恢复的联邦检索任务中，固定 Top-3 是更可靠的 RAGRoute 配置。该发现也为后续动态预算方法提供了明确要求：若要声称自适应路由优越，必须在同样严格的独立冻结测试上证明其成本下降不会系统性损害完整证据恢复。

## 8. 执行事件与修复记录

盲态路由执行期间出现三类配置错误，均在写入任何有效上下文前停止；每次仅删除 0 行的失败输出后重启，不修改冻结样本、模型权重、候选集、阈值、Reader 或统计规则：

1. 历史 source centroids 的根路径最初指向项目根目录，后修正为冻结实验根目录。
2. 初始代码将全局 20 个 source centroids 误当作每条查询的 8 个 P0 候选；后明确使用 20-source 表示，而只对冻结 Top-8 的客户端打分。
3. canonical SQLite indexes 和 HuggingFace Reader cache 初始路径不正确；均修正为冻结实验根目录下既有的只读产物。

这些修复仅恢复预先规定的资源定位，不改变实验方法。相关实现已同步至 GitHub，提交链末端为 `e96e4d8`。

## 9. 审计范围与限制

1. B3F 仅比较 RAGRoute 的 Fixed3 与 OriginalThreshold，不能直接替代 C1/C2 中包含 M2、B0、B1、B4a、B6 等方法的全基线比较。
2. `OriginalThreshold` 的客户端数是模型输出结果而非固定同预算设置；本报告将其作为通信--效用权衡分析，而不将其宣称为严格同预算优胜。
3. 为守住解封前标签防火墙，Reader 没有生成 support predictions；因此没有 `SP F1` 和 `Joint F1`。答案 F1 和证据完整性指标仍可审计，但不应外推到完整 joint QA 指标。
4. 置信区间是配对 bootstrap CI；本报告不作未经预先规定的多重比较显著性声明。

## 10. 推荐后续工作

1. 将 `Fixed3` 作为 RAGRoute 的稳健主对照，并把 OriginalThreshold 明确放在低通信、非同预算参考组。
2. 在独立冻结样本上增加预算匹配的 RAGRoute 变体，例如固定 1、2、3 客户端的概率排序规则，绘制通信--证据完整率前沿。
3. 若需要报告 `SP F1` 与 `Joint F1`，应在新的预注册实验中预先定义不读取 gold 的 support prediction 机制，而不要回写或修改本次 B3F 盲态产物。
4. 与 M2、静态 Top-3、随机 Top-3 和全候选高成本参考在同一冻结集上进行统一比较，才能支持方法级论文结论。
