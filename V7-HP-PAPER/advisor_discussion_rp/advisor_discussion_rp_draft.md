# Research Proposal Draft for Advisor Discussion

任务名：`V7-HP-PAPER-advisor_discussion_RP_story_version`  
研究方向：Federated RAG / Multi-hop QA / Reader-safe Context Selection  
用途：和导师讨论研究 idea、论文定位与下一步实验计划

## 0. Candidate Titles

候选标题 1：

**Federated RAG 中面向多跳问答的 Answer-Neutral Evidence Selection**

优点是直观，能看出任务场景是 Federated RAG 和多跳问答，也能看出方法核心是 answer-neutral evidence selection。缺点是 “evidence selection” 可能让人误解为普通检索重排，而不是 routing action 到 reader context 的转换问题。

候选标题 2：

**从 Federated Routing 到 Reader-Safe Context Selection：多跳问答中的证据调度问题**

优点是适合导师讨论，问题意识比较清楚：我不是只做 retrieval，也不是只做 reader，而是在研究 routing 后的 context selection。缺点是标题偏中文讨论版，正式投稿时可能需要英文重写。

候选标题 3：

**Bridging the Policy-Action-to-Reader Gap in Federated RAG for Multi-Hop QA**

优点是最接近论文贡献，强调本文提出的核心概念 `policy-action-to-reader gap`。缺点是导师如果不熟悉该领域，第一次读可能需要解释。

**推荐标题：**

**从 Federated Routing 到 Reader-Safe Context Selection：多跳问答中的证据调度问题**

推荐理由：这版最适合和导师讨论。它没有过度承诺“解决 Federated RAG”，也没有把结论写得太技术化；同时能自然引出本文真正的问题：上游 routing 找到的候选证据，是否应该进入 reader 的输入上下文。

## 1. One-sentence Summary

本文研究的问题是：在 Federated RAG 的多跳问答场景中，上游 federated routing 找到的候选证据或 context action 不应被直接交给 reader，而需要一个 `answer-neutral positive-action selector` 判断它是否既能帮助证据链、又不会损害最终答案。

**小结：** 这项研究不是单纯改进检索器，也不是单纯改进 reader，而是研究两者之间容易被忽略的“证据调度层”。

## 2. Motivation: Why This Problem Matters

大语言模型虽然有很强的语言生成能力，但在知识密集型任务中仍然容易出现事实错误、知识过时或无法解释的问题。RAG，即 Retrieval-Augmented Generation，通常译作检索增强生成，试图解决这个问题：系统先从外部知识库中检索相关资料，再让模型基于这些资料回答问题。直观地说，RAG 像是给模型发一份参考资料，让它不要只凭记忆作答。

但在真实应用中，资料往往不是集中存放的。医院、学校、公司、研究组、用户设备都可能有自己的私有知识库。由于隐私、权限、通信成本或数据所有权问题，我们不能简单把所有资料集中到一个服务器。这就引出 Federated RAG：在数据分布于多个客户端或机构的情况下，系统需要从不同来源中找到可能有用的信息，但又不能把所有数据直接汇总。

如果只是普通问答，这个问题已经有难度；如果是多跳问答，难度会进一步放大。所谓 multi-hop QA，指的是回答一个问题需要连接多个证据。比如：

> 问：某位作家的出生城市所属国家的首都是哪里？

模型可能需要先找到作家的出生城市，再判断这个城市属于哪个国家，最后再查这个国家的首都。这里不是找到一段相关资料就够了，而是要构造一条证据链。

HotpotQA 这类数据集正是为了评估这种能力。它不只看最终答案是否正确，还看模型是否找到了 supporting facts，也就是支持答案的关键证据，并进一步看 joint_f1，即答案和证据是否同时正确。换句话说，多跳问答中的成功不只是“答对”，也包括“有正确的推理证据”。

一个自然想法是：既然多跳问答需要证据，那是不是检索到越多相关证据，reader 就答得越好？实验和直觉都告诉我们，答案是否定的。reader 可以理解为考试学生，context 可以理解为考场发给他的参考资料。资料太少当然不行，但资料太多、顺序混乱、夹杂干扰材料，或者把原本最关键的答案线索替换掉，也会让学生答错。

这就是本文关注的问题：federated routing 在上游可能找到一些看起来和 supporting evidence 相关的内容，但这些内容不能被无条件塞进 reader context。某些 support-like context 可能提升 support recall，却降低 answer_f1。也就是说：

```text
routing-side support gain ≠ reader-side answer / joint gain
```

我把这个现象称为：

```text
policy-action-to-reader gap
```

这里的 policy 指上游 routing policy，action 指它产生的具体动作，例如插入一段证据、替换一段资料、重排 context；reader 指最终阅读 context 并回答问题的模型。这个 gap 的意思是：上游 policy 认为有用的 action，不一定真的能让 reader 答得更好。

**小结：** 这项研究的动机来自一个实际但容易被忽略的问题：在 Federated RAG 中，找到候选证据只是第一步，如何把候选证据安全地交给 reader，才是多跳问答中决定最终效果的关键。

## 3. The Story Behind the Idea

这个研究 idea 不是一开始就直接指向 answer-neutral selector，而是从 federated routing 的实验中逐渐演化出来的。

最初，我关注的是 Federated RAG 中的 selective routing / selective upload 问题。因为数据分散在不同 clients 上，通信预算又有限，不可能让所有客户端上传全部信息。因此，我希望设计更智能的 federated routing、hypernet 或 agent policy，让客户端只上传更有价值的参数块、证据或 routing signal。

这一阶段的核心直觉是：如果系统能够在有限通信预算下选出更有价值的信息，就应该能改善下游 RAG。为此，我尝试了 HP-hyper、BSP-DIAG 以及多个 agent policy 变体，包括 memory、bandit、slot 等设计。这些方法的共同目标是让客户端更聪明地选择上传内容，而不是固定上传或随机上传。

从通信和检索角度看，这条路线不是没有价值。HP-hyper 的诊断结果显示，在 same-payload 约束下，系统可以维持 retrieval performance；在 hard subset 上，V6/HP-hyper 也有很小的 retrieval-side positive signal。pooler / layerwise ablation 进一步说明，policy 确实能改变 selected block distribution。也就是说，系统不是完全没有学到选择行为。

但问题出现在下游。尽管 routing 或 parameter-block selection 行为发生了变化，这些差异在 retrieval / reader pipeline 中经常被压平。更具体地说，aggregation、dense embedding 表示、top-k retrieval 和 reader context construction 会削弱上游 policy 的差异。BSP-DIAG 的 official FiD/T5 reader 结果也显示，不同 agent memory / bandit / slot 变体的 answer_f1、sp_f1、joint_f1 几乎打平。

这组结果一开始看起来像负结果，但我认为它更像是定位问题的诊断证据。它说明：更复杂的 federated routing policy 可以改变上传或检索行为，但这些变化不会自动变成 reader-side QA gain。问题不只在“能否找到更多候选证据”，还在“这些候选证据是否应该进入 reader context”。

因此，研究问题发生了转向。与其继续无限增强上游 routing policy，不如显式研究 routing action 到 reader context 之间的转换问题。也就是说，当 federated routing 产生一个候选 action 后，我们需要判断：这个 action 是真的应该被应用，还是应该被拒绝？

这个转向也是当前论文主线的来源。HP-hyper / BSP-DIAG 不再作为主方法成功结果，而是作为 motivation 和 diagnostic evidence：它们告诉我们，单纯强化 routing policy 不够；需要一个下游 reader-safe action selection 机制。

**小结：** 前期 HP-hyper 和 BSP-DIAG 的价值在于帮助我从“如何设计更复杂 routing policy”转向“如何判断 routing action 是否 reader-safe”。这个转向使研究问题更清楚，也更接近论文贡献。

## 4. Key Research Question

本文的核心研究问题可以写成：

```text
Given a baseline reader context and a candidate context action produced after federated routing,
how can we decide whether this action should be applied to the reader context?
```

中文解释是：

给定一个原始 reader context，以及 federated routing 产生的一个候选动作，例如插入一段证据、替换一段资料、重排上下文，我们如何判断这个动作是否值得应用？

这个问题的关键不是单纯判断 evidence 是否相关，而是要同时判断三件事：

1. 它是否可能改善 supporting evidence；
2. 它是否不会破坏 answer quality；
3. 它是否能提升 joint_f1，即答案和证据一起变好。

为了表达这个问题，可以用一个简化定义：

```text
Positive action: joint/support utility improves.
Answer-neutral action: answer quality does not decrease.
Target action: positive + answer-neutral.
```

如果一个 action 提高了 support recall，但让 answer_f1 下降，它就不应该被视为成功 action。相反，本文真正关心的是那些在保护 answer 的前提下改善 support / joint 的 action。

**小结：** 本文的研究问题不是“如何找到更多证据”，而是“如何判断一个候选证据动作是否应该进入 reader context”。

## 5. Proposed Method

我提出的方法叫：

```text
answer-neutral positive-action selector
```

可以把它理解成一个“谨慎的证据调度员”。它不是一个“尽可能多找资料的人”，而是一个“考试前帮学生整理资料的人”。它的目标不是把所有资料塞进书包，而是保留会帮助推理、同时不会干扰答题的资料。

具体来说，selector 的输入包括：

- baseline reader context，即原始 reader 会看到的资料；
- candidate context action，即 federated routing 产生的候选动作；
- 一些 no-leak features，例如 query-document lexical overlap、entity overlap、retrieval score、rank、context position、candidate action 的类型等。

selector 要判断：

- 这个 action 是否可能改善 support / evidence；
- 这个 action 是否可能保持 answer_f1 不下降；
- 这个 action 是否可能提升 joint_f1。

selector 的输出很简单：

- apply：应用这个 action，把候选 context 修改交给 reader；
- reject / fallback：拒绝这个 action，保留 baseline reader context。

这里最重要的是 answer-neutral。它并不是追求 answer_f1 大幅提升，而是先确保 answer quality 不下降。在保护 answer 的基础上，再追求 support / joint 改善。因此，论文中更准确的表述是 preserving answer_f1，而不是 significantly improving answer_f1。

另一个关键点是 no-leak。因为本文的标签与 reader outcome 有关，如果不严格控制数据泄漏，很容易把“事后知道这个 action 有用”误写成“模型真的能提前判断这个 action 有用”。因此，实验采用 strict no-leak query-level cross-fitting：

- 某个 query 的 reader outcome 不能用于训练该 query 的 selector；
- gold answer / gold support 不能作为 inference-time features；
- oracle 只能作为诊断上限，不能作为正式方法；
- selector 只能用 train folds 学到的规律去判断 held-out queries。

这个设置保证了方法更接近真实推理场景。selector 在测试时不能偷看答案，只能根据可用特征判断一个 action 是否安全、有用。

**小结：** answer-neutral positive-action selector 的核心思想是谨慎地应用 routing action：只有当 action 被预测为不伤害答案、并可能改善 support/joint 时，才让它进入 reader context。

## 6. Preliminary Experiments

### 6.1 HP-hyper / BSP-DIAG Diagnostic Evidence

HP-hyper 和 BSP-DIAG 的作用是帮助定位问题，而不是作为主方法成功结果。

HP-hyper 显示，在 same-payload 约束下，selective upload / parameter-block selection 能够维持 retrieval performance。在 hard subset 上，V6/HP-hyper 有很小的 retrieval-side positive signal。pooler / layerwise ablation 也说明，policy 确实能改变 selected block distribution。

但这些变化并没有自然转化为 reader-side QA gain。pooler / layerwise 变化带来的 performance changes 很小，甚至有时只是改变选择行为而不改善下游效果。BSP-DIAG official FiD/T5 reader 结果也显示，不同 agent memory / bandit / slot 变体在 answer_f1、sp_f1、joint_f1 上几乎打平。

这说明 routing-side policy differences are flattened downstream。也就是说，上游 policy 的差异会被 aggregation、dense embedding、top-k retrieval 和 reader context selection 压平。因此，仅靠继续设计更复杂的 routing policy，很难自然得到 reader-side QA gain。

**小结：** HP-hyper / BSP-DIAG 是关键诊断证据：它们说明问题不只是 routing policy 不够强，而是缺少从 routing action 到 reader context 的安全选择机制。

### 6.2 HotpotQA Main Result

当前主结果来自 HotpotQA 上的 V7-HP-PAPER v2.3，即 `answer-neutral positive-action selector`。

关键结果如下：

- support_recall@5 提升约 `+0.0190`；
- sp_f1 提升约 `+0.0254`；
- joint_f1 提升约 `+0.0150`，并具有统计显著性；
- answer_f1 有小幅正向变化 `+0.0023`，但不显著；
- fallback_rate 为 `0.5000`；
- positive_candidate_recall 为 `0.3288`。

这个结果说明，answer-neutral selection 可以把 routing-side support signal 转化为 reader-side joint/support gain。更准确的论文表述应是：

> The selector significantly improves joint_f1 and support-side metrics while preserving answer_f1.

不能写成：

> The selector significantly improves answer_f1.

因为 answer_f1 的提升很小且不显著。这里的贡献不是让 reader 答案能力大幅变强，而是在不伤害 answer quality 的前提下，让 support 和 joint 指标显著改善。

**小结：** HotpotQA 主结果支持本文方法：answer-neutral selector 能把原本不稳定的 routing-side signal 转化为更可靠的 reader-side joint/support gain。

### 6.3 2Wiki External Diagnostic

2WikiMultiHopQA 实验应该作为 external diagnostic 或 limitation，而不是成功泛化结果。

2Wiki 的价值在于，它说明 adapter 和 reader-backed evaluation pipeline 可以迁移到另一个 multi-hop QA 数据集。但在强 BM25 baseline 下，selector-level generalization 没有可靠成立。进一步分析显示，问题包括：

- candidate exposure 不足；
- positive actions 不容易被当前 features 区分；
- safety predictor 跨数据集校准弱；
- BM25 baseline 已经很强。

因此，2Wiki 让论文更诚实：本文方法在 HotpotQA 上成立，但跨数据集泛化仍需要更强的 candidate generation 和 dataset-specific safety calibration。它不应该被写成“方法成功泛化到 2Wiki”，而应被写成外部诊断和 limitation。

**小结：** 2Wiki 的作用是给出方法边界：当前 selector 的主证据在 HotpotQA，跨数据集迁移仍有挑战。

## 7. Expected Contribution

本文预期贡献可以概括为四点。

第一，提出并明确了 Federated RAG 多跳问答中的 `policy-action-to-reader gap`：上游 routing policy 产生的 action 不一定能转化为 reader-side QA gain。

第二，将问题从单纯 federated routing / selective upload 转化为 reader-safe context action selection，强调下游 context action 是否应该应用。

第三，提出 `answer-neutral positive-action selector`，在 strict no-leak query-level cross-fitting 下，只选择预测为 answer-safe 且 joint/support beneficial 的 action。

第四，在 HotpotQA 上验证该方向有效：方法显著提升 joint_f1、support_recall@5 和 sp_f1，同时保持 answer_f1；同时通过 HP-hyper、BSP-DIAG 和 2Wiki 诊断明确方法边界。

**小结：** 本文贡献不是声称解决 Federated RAG，而是提出一个更具体、更可验证的问题：如何把 federated routing 产生的候选 action 安全地转化为 reader-side gain。

## 8. Current Limitations

当前研究仍有几个明显限制。

第一，主结果集中在 HotpotQA。虽然 HotpotQA 是合理的多跳问答基准，但这意味着论文应被定位为 HotpotQA-centered paper with diagnostic limitations，而不是跨所有数据集的通用结论。

第二，answer_f1 不是显著提升。当前 answer_f1 是 small non-significant positive delta，因此应写作 answer-preserving。论文不能声称显著提升 answer_f1。

第三，2Wiki 不是成功泛化。它说明 pipeline 可以迁移，但 selector-level generalization 仍受 candidate exposure、feature separability 和 safety calibration 限制。

第四，multi-reader replication 仍受模型环境限制。reader sensitivity 是本文的重要问题之一，因此如果资源允许，补充多 reader replication 会增强论文说服力。

第五，candidate generation 仍是未来工作。selector 的上限取决于候选 action pool。如果候选池中缺乏真正 positive 且 answer-safe 的 action，再好的 selector 也无法创造收益。

**小结：** 当前论文应诚实定位为：在 HotpotQA 上验证 answer-neutral action selection 的有效性，并通过诊断实验说明问题边界，而不是宣称解决所有 Federated RAG 或跨数据集泛化。

## 9. Discussion Points for Advisor

我希望和导师重点讨论以下问题：

1. 这个问题定位是否足够清楚：从 federated routing 转向 reader-safe context selection 是否合理？
2. `policy-action-to-reader gap` 这个概念是否适合作为论文核心 framing？
3. HP-hyper / BSP-DIAG 是否应放在 motivation / diagnostic section，而不是主结果？
4. 论文标题应更偏 Federated RAG，还是更偏 Reader-safe Context Selection？
5. 当前 HotpotQA v2.3 结果是否足够支撑一篇投稿？
6. 目标会议应考虑 EMNLP/NAACL Findings、COLING，还是尝试 main conference？
7. 是否需要补 multi-reader replication 来增强 reader sensitivity 相关论证？
8. 2Wiki 应该放在 limitation、appendix，还是 external diagnostic section？
9. 是否需要进一步强化理论表述，例如把 action selection 写成 risk-controlled decision problem？
10. 在论文摘要中应如何控制 claim，避免 reviewer 认为我们夸大 answer_f1 或 cross-dataset generalization？

**小结：** 这些问题主要围绕论文定位、claim 强度和补实验优先级，适合导师帮助判断投稿策略。

## 10. Next Steps

下一步计划如下。

第一，完成 Introduction / Background / Problem Formulation，把研究故事写清楚，尤其是 Federated RAG、multi-hop QA、policy-action-to-reader gap 和 answer-neutral selection 之间的逻辑关系。

第二，整理主结果表和 ablation 表。主表放 HotpotQA v2.3，强调 joint/support 显著提升和 answer_f1 preserved；ablation 表放 v2.2、no answer constraint、no safety predictor 等对照。

第三，将 HP-hyper 写入 diagnostic section。它的作用是说明 routing policy differences can be flattened downstream，从而动机化 downstream action selection。

第四，将 2Wiki 写入 limitation 或 appendix。它可用于说明 pipeline 可迁移，但 selector-level cross-dataset generalization 尚未成立。

第五，根据导师反馈决定是否补 multi-reader replication。如果导师认为 reviewer 会强烈质疑 reader-specific 结果，应优先补一个额外 reader；如果资源不足，则在 limitation 中明确说明。

第六，准备一页纸版本和 claim boundary checklist，用于会前快速沟通，确保讨论聚焦于研究定位而不是实验细节。

**小结：** 当前最重要的不是继续盲目加实验，而是先确定论文主线是否成立：Federated routing 产生候选 action，answer-neutral selector 决定哪些 action 可以安全进入 reader context。

## 11. Final Positioning

这不是一篇声称“解决 Federated RAG”的论文，也不是一篇声称“跨所有数据集泛化成功”的论文。更准确的定位是：

> 本文研究 Federated RAG 中 routing action 到 reader context 之间的转换问题。我们指出 routing-side support gain 不会自动变成 reader-side QA gain，并提出 answer-neutral positive-action selector，在 HotpotQA 上显著提升 joint/support metrics，同时保持 answer_f1。HP-hyper 和 2Wiki 诊断进一步说明，这一问题真实存在，且跨数据集泛化仍有挑战。

这个定位的优点是：claim 清楚、边界诚实、方法有新意，也能解释前期看似负面的 HP-hyper / BSP-DIAG 结果为什么对论文仍然有价值。

**小结：** 当前论文最有希望的主线不是“更强 routing policy”，而是“reader-safe action selection bridges federated routing and downstream multi-hop QA”。

## 12. Self-check

```text
ready_for_advisor_discussion: true
main_story_clear: true
technical_terms_explained: true
claim_boundary_respected: true
recommended_next_action: 先将本 RP 发给导师确认论文定位，再根据反馈决定是否补 multi-reader replication 或调整 HP-hyper/2Wiki 的论文位置。
```
