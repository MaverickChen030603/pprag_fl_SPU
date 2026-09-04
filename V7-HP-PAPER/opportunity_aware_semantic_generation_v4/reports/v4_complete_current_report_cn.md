# V7-HP-PAPER V4 当前完整实验报告

## 1. 研究目的

V4 不再继续堆固定 action templates，也不再把 selector 当作第一瓶颈。它验证一个更基础的问题：对于 v2/v3 根本没有正动作的 query，能否通过语义互补、query-conditioned 的候选构造，主动创造 reader-compatible opportunity。

V2 是冻结投稿 fallback；v3 是 negative opportunity study。V4 独立运行，没有覆盖两者。

## 2. 冻结事实与重新审计

- v2 主可用动作：4,000；positive density 9.48%；positive-query coverage 20.3%。
- v3 有效动作：7,882；positive density 9.43%；coverage 23.4%。
- 按 `baseline answer_f1=1 且 title_recall=1` 精确重算，ceiling query 为 389，non-ceiling 为 611，v3 conditional coverage 为 38.3%。
- v3 相比 v2 净增 31 个 covered queries，但集合层面是新增 81 个、同时未恢复 50 个 v2-positive queries。论文必须同时报告净增与 marginal new coverage。

## 3. V4 方法

V4 使用五折 fully nested protocol。每折仅用 outer-train 的 v3 reader outcomes 训练：

1. missing-hop estimator；
2. MPNet bi-encoder + MS MARCO cross-encoder 特征上的 semantic document opportunity model；
3. two-document pair complementarity model。

outer-test 生成时冻结所有组件，只读取 question、baseline context、document text、BM25/semantic scores、非 gold entity 与 lexical relation。gold answer、gold support、target reader outcome、oracle action 均未使用。生成器最多给每题 8 个动作，覆盖 complementary insertion、anchor-preserving replacement、semantic two-doc chain、redundancy replacement 和两类顺序动作。

模型诊断：document opportunity inner-CV AP 均值 **0.2308**，pair complementarity AP 均值 **0.3274**。这高于各自正类基率，但 missing-hop 的稀有 ordering/redundancy 类仍难学习。

## 4. Action Opportunity 结果

- Queries: **1000**。
- Effective actions: **7,934**；其中相对 v3 新 context actions **5,655**。
- Positive actions: **1,167**；density **14.71%**，较 v3 9.43% 提升 **+5.28%**。
- Positive-query coverage: **29.2%**，较 v3 提升 **+5.8%**，但距 30% gate 差 0.8 点。
- Non-ceiling coverage: **47.63%**。
- 新覆盖 v3 未覆盖 query: **81**。
- Answer-safe action rate: **92.66%**。
- New-query efficiency: **0.0143**，低于冻结强化门槛。

| Opportunity gate | Result |
| --- | --- |
| A_overall_opportunity | FAIL |
| B_conditional_opportunity | PASS |
| C_marginal_breadth | PASS |
| D_action_quality_density | PASS |
| E_new_query_efficiency | FAIL |

结果为 **3/5**：没有达到 4/5 strong pass，但只失败 2 项，因此未触发原始“至少 4 项失败才停止”的 mandatory stop，后续 selector 被标记为 borderline continuation。

## 5. Fully Nested Selector

The fully nested selector intervenes on 260/1000 queries and improves answer F1 by +0.0133 (95% CI [+0.0024, +0.0249], p=0.0176), title recall by +0.0455, and the answer-title product by +0.0442.

Selector 选择覆盖率为 **26.0%**，selected-action answer-drop risk 为 **5.0%**。最关键的是 answer F1 从 v2/v3 的非显著小增益，变成了本轮显著正增益；但这仍是同一 1,000-query development protocol，不能替代 scale-up。

## 6. Official HotpotQA

Under official sentence-level evaluation, answer F1 changes by +0.0133 (p=0.0176), supporting-fact F1 by +0.0053 (p=0.0372), and joint F1 by +0.0064 (p=0.0752).

官方指标边界必须明确：answer F1 与 supporting-fact F1 分别显著为正；joint F1 是正向趋势，`p=0.0752`，不得写成显著。Title recall/F1 仍是诊断 proxy，未被重命名为 official supporting-fact metrics。

## 7. Multi-Reader

UnifiedQA confirms the direction with answer F1 +0.0129 and joint F1 +0.0088; its answer-drop rate is 1.5%.

FLAN 与 UnifiedQA 的 answer/joint 方向一致，支持“不是单一 reader 偶然现象”；但当前第二 reader 没有独立重新训练 support predictor，support 部分复用相同 frozen context 与 nested sentence predictor。

## 8. 同源 3,000-query Frozen Scale-Up

On 3,000 disjoint same-source queries, the unchanged selector intervenes on 774/3000 queries. FLAN answer F1, supporting-fact F1, and joint F1 improve by +0.0088 (p=0.0096), +0.0056 (p=0.0004), and +0.0064 (p=0.0004); UnifiedQA yields answer/joint gains of +0.0110/+0.0085.


| Reader | N | Answer F1 baseline | Answer F1 selected | Delta | SP F1 delta | Joint F1 delta | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | 3000 | 0.6183 | 0.6271 | +0.0088 | +0.0056 | +0.0064 | 0.0004 |
| UnifiedQA-T5-Large | 3000 | 0.5662 | 0.5772 | +0.0110 | +0.0056 | +0.0085 | <0.0002 |


Scale-up 使用 `hotpot_qa/distractor/validation`、seed 44 的固定顺序，取原开发 1,000 之后的 3,000 条互斥 query。原 1,000 source 与 HybridSoftRetriever baseline title-order 均达到 100% 复现；未使用临时 BM25-only top-5。Generator、selector threshold/coverage、reader prompt/decoding 与 sentence-support threshold 均未在 3,000 条上调参。

该结果把 1,000-query 上 joint F1 的非显著正趋势推进为同源 scale-up 上的显著正增益，并且两个 reader 的 answer/joint 方向一致。Joint EM 的置信区间仍跨 0，不应宣称显著。

## 9. 当前论文判断

最终状态：**main_conference_stretch**。

可以主张：

- semantic generation 明显提高 positive-action density 与 query opportunity；
- fully nested selector 在 1,000-query HotpotQA 上得到显著 answer F1 与 SP F1 正增益；
- 3,000 条互斥同源 query 在不调参条件下复现 answer/SP/joint F1 正增益；
- 双 reader 方向一致，无系统性 answer degradation。

不能主张：

- opportunity 五项全面通过；
- 1,000-query development 上 official joint F1 已显著；
- 3,000-query 同源 scale-up 等价于跨数据集泛化；
- 2Wiki/MuSiQue 已验证；
- SOTA 或 faithful external-method comparison。

## 10. 未完成项与下一步

Scale-up 当前状态为 `complete`。主表现在可以加入 3,000-query 同源冻结复现，但必须与 1,000-query development 结果分行报告，并明确 baseline 是 `HybridSoftRetriever(alpha=0.55, uniform weights, top_k<=5)`，不是 BM25-only top-5。

下一步不再调整 HotpotQA scale-up 参数，优先完成预注册的 external dataset validation 与 faithful external-method comparison。只有外部数据集也保持方向，才能把“同源规模稳健性”升级为“跨数据集泛化”。
