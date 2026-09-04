# One-page Summary for Advisor

## Proposed Title

**从 Federated Routing 到 Reader-Safe Context Selection：多跳问答中的证据调度问题**

## One-sentence Summary

本研究关注 Federated RAG 多跳问答中的一个中间层问题：上游 federated routing 找到的候选证据或 context action，不应被直接交给 reader，而需要判断它是否既能帮助证据链、又不会损害最终答案。

## Motivation

RAG，即 Retrieval-Augmented Generation，通过“先检索资料、再让模型阅读资料回答问题”来提升事实性。但在真实场景中，知识往往分散在不同机构、用户或客户端，不能全部集中到一个服务器，因此需要 Federated RAG。

多跳问答比普通问答更难。它不只需要找到一段相关资料，而是需要连接多段 evidence 形成推理链。HotpotQA 这类数据集不仅评估最终 answer，还评估 supporting facts 和 joint_f1。

核心问题是：找到更多 support-like evidence 并不一定让 reader 答得更好。reader 像考试学生，context 像参考资料；资料太少不行，但资料太多、顺序混乱、夹杂干扰材料，或者替换掉关键答案线索，也会让学生答错。

因此本文提出一个核心 gap：

```text
routing-side support gain ≠ reader-side answer / joint gain
```

我将它称为 `policy-action-to-reader gap`。

## Story Behind the Idea

我最初尝试通过更复杂的 federated routing / selective upload / hypernet / agent policy，让客户端在通信预算有限的情况下选择更有价值的参数块、证据或 routing signal。

HP-hyper 和 BSP-DIAG 的诊断结果显示：

- HP-hyper 在 same-payload 约束下能维持 retrieval performance；
- hard subset 上有很小的 retrieval-side positive signal；
- pooler / layerwise ablation 说明 policy 确实能改变 selected block distribution；
- 但这些差异在 aggregation、dense embedding、top-k retrieval 和 reader context construction 中被压平；
- BSP-DIAG official reader 结果显示，不同 agent memory / bandit / slot 变体的 official QA 指标几乎打平。

这些结果不是失败，而是定位了真正问题：复杂 routing policy 可以改变上传或检索行为，但不会自动变成 reader-side QA gain。因此研究问题从“如何设计更强 routing policy”转向“如何判断 routed context action 是否应该进入 reader context”。

## Research Question

```text
Given a candidate action after federated routing, should we apply it to the reader context?
```

中文解释：给定 baseline reader context 和 federated routing 产生的候选动作，例如插入、替换或重排一段 context，如何判断这个动作是否值得应用？

判断标准不是单纯 evidence 是否相关，而是：

1. 是否改善 supporting evidence；
2. 是否不会破坏 answer quality；
3. 是否提升 joint_f1。

## Proposed Method

方法名称：`answer-neutral positive-action selector`

可以理解为“谨慎的证据调度员”。它不是尽可能多找资料，而是帮 reader 整理资料：只保留会帮助推理、同时不会干扰答题的 context action。

它只选择预测为：

- 不会损害最终答案；
- 能改善证据链；
- 能提升 joint/support utility；

的 action。

其中 `answer-neutral` 的含义是：本文不声称显著提升 answer_f1，而是先确保 answer quality 不下降，在此基础上提升 support / joint metrics。

实验采用 strict no-leak query-level cross-fitting，避免 selector 在测试时偷看 gold answer、gold support 或该 query 的 reader outcome。

## Preliminary Results

HotpotQA v2.3 是当前主结果：

- `support_recall@5` 提升约 `+0.0190`
- `sp_f1` 提升约 `+0.0254`
- `joint_f1` 提升约 `+0.0150`，具有统计显著性
- `answer_f1` 小幅正向 `+0.0023`，但不显著

准确表述应为：

> The selector significantly improves joint_f1 and support-side metrics while preserving answer_f1.

2WikiMultiHopQA 作为 external diagnostic：pipeline 可以迁移到另一个 multi-hop QA 数据集，但 selector-level generalization 尚未可靠成立，主要受 candidate exposure、feature separability、safety calibration 和强 BM25 baseline 限制。

## Expected Contribution

1. 提出 Federated RAG 多跳问答中的 `policy-action-to-reader gap`。
2. 将问题从 federated routing 转化为 reader-safe context action selection。
3. 提出 `answer-neutral positive-action selector`，在 no-leak 条件下选择 answer-safe 且 joint/support beneficial 的 action。
4. 在 HotpotQA 上显著提升 joint/support metrics，同时保持 answer_f1，并通过 HP-hyper / 2Wiki 诊断明确方法边界。

## Current Limitations

- 主结果集中在 HotpotQA；
- answer_f1 不是显著提升，只能写 answer-preserving；
- 2Wiki 不是成功泛化；
- multi-reader replication 仍受模型环境限制；
- candidate generation 仍是未来工作。

## Advisor Discussion Focus

希望导师帮助判断：

1. 这个问题定位是否清楚；
2. HP-hyper 是否应放在 motivation / diagnostic 而非主结果；
3. 标题应偏 Federated RAG 还是 Reader-safe Context Selection；
4. 当前结果是否足够支撑投稿；
5. 是否需要补 multi-reader replication；
6. 2Wiki 应放 limitation 还是 appendix。

## Self-check

```text
ready_for_advisor_discussion: true
main_story_clear: true
technical_terms_explained: true
claim_boundary_respected: true
recommended_next_action: 用这份一页纸先和导师确认论文定位，再决定是否补实验。
```
