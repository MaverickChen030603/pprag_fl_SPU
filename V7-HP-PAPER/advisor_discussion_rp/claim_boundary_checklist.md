# Claim Boundary Checklist

## Recommended Paper Positioning

本文适合定位为：

> A HotpotQA-centered study on answer-neutral action selection after federated routing, with HP-hyper and 2Wiki diagnostics clarifying the boundary of the method.

中文表述：

> 本文研究 Federated RAG 中 routing action 到 reader context 之间的转换问题。我们指出 routing-side support gain 不会自动变成 reader-side QA gain，并提出 answer-neutral positive-action selector，在 HotpotQA 上显著提升 joint/support metrics，同时保持 answer_f1。

## Claims We Can Make

1. **可以说：Federated routing / HP-hyper 能改变 routing 或 selection behavior。**

   依据：pooler / layerwise ablation 显示 selected block distribution 可以被改变。

2. **可以说：HP-hyper 在 same-payload 约束下可以维持 retrieval performance。**

   依据：payload 约束下检索指标没有明显崩塌。

3. **可以说：HP-hyper 在 hard subsets 上有很小的 retrieval-side positive signal。**

   注意：只能说 small / directional retrieval-side signal，不能说 significant QA gain。

4. **可以说：routing-side policy differences can be flattened downstream。**

   依据：HP-hyper / BSP-DIAG 显示，policy 行为差异经过 aggregation、dense embedding、top-k retrieval 和 reader context construction 后，reader-side 指标几乎打平。

5. **可以说：这说明需要 downstream action selection。**

   这是本文从 routing policy 转向 answer-neutral selector 的核心动机。

6. **可以说：V7-HP-PAPER v2.3 在 HotpotQA 上显著提升 joint_f1。**

   依据：`joint_f1_delta = +0.0150`，统计显著。

7. **可以说：V7-HP-PAPER v2.3 显著提升 support_recall@5 和 sp_f1。**

   依据：
   - `support_recall@5_delta = +0.0190`
   - `sp_f1_delta = +0.0254`

8. **可以说：V7-HP-PAPER v2.3 preserves answer_f1。**

   依据：`answer_f1_delta = +0.0023`，小幅正向但不显著。推荐写 preserving，而不是 improving。

9. **可以说：2Wiki verifies adapter and reader-backed evaluation pipeline transfer。**

   解释：2Wiki 说明 pipeline 可以迁移到另一个 multi-hop QA 数据集。

10. **可以说：2Wiki diagnostics show cross-dataset selector transfer remains limited。**

    依据：在强 BM25 baseline 下，selector-level generalization 没有可靠成立，受 candidate exposure、feature separability 和 safety calibration 限制。

## Claims We Should Not Make

1. **不能说：HP-hyper significantly improves official QA。**

   HP-hyper 是 diagnostic evidence，不是主方法成功结果。

2. **不能说：agent memory / bandit / slot policy directly improves reader performance。**

   BSP-DIAG official reader groups 几乎打平。

3. **不能说：answer_f1 significantly improves。**

   answer_f1 只有 small non-significant positive delta。

4. **不能说：the method solves Federated RAG。**

   本文解决的是更具体的 downstream action-selection problem，不是整个 Federated RAG。

5. **不能说：the method successfully generalizes to 2Wiki。**

   2Wiki 只能作为 external diagnostic / limitation。

6. **不能说：the method universally works for all readers。**

   multi-reader replication 仍有限，reader sensitivity 仍是 limitation。

7. **不能说：oracle selector is an inference-time method。**

   oracle 只能作为 diagnostic upper bound。

8. **不能说：the method reaches SOTA。**

   当前目标不是 SOTA claim，而是验证 answer-neutral action selection 的研究问题。

9. **不能说：support gain always improves answer generation。**

   本文正是指出 support-like evidence 可能伤害 answer quality。

10. **不能说：more complex routing policy is sufficient。**

    HP-hyper / BSP-DIAG 的核心结论恰恰是复杂 routing policy 不会自动转化为 reader gain。

## Safe Abstract Claim

推荐摘要中的安全表述：

> We study the downstream action-selection problem after federated routing in multi-hop RAG. Our diagnostics show that routing-side support signals do not automatically translate into reader-side QA gains. We propose an answer-neutral positive-action selector that applies a routed context action only when it is predicted to preserve answer quality and improve joint/support utility. On HotpotQA, the selector significantly improves joint F1 and support-side metrics while preserving answer F1.

## Unsafe Abstract Claims

应避免：

```text
Our method significantly improves answer F1.
Our federated routing agent solves HotpotQA.
Our method generalizes to 2Wiki.
Our method achieves SOTA.
Our oracle selector can be used at inference time.
```

## Recommended Terminology

优先使用：

- `answer-neutral`
- `answer-preserving`
- `joint/support improvement`
- `strict no-leak`
- `query-level cross-fitting`
- `external diagnostic`
- `mechanism diagnostic`
- `claim boundary`

谨慎使用：

- `answer-improving`
- `generalization`
- `solves`
- `SOTA`
- `universal`

## Self-check

```text
ready_for_advisor_discussion: true
main_story_clear: true
technical_terms_explained: true
claim_boundary_respected: true
recommended_next_action: 写论文摘要和 Introduction 时始终对照本 checklist，避免把 diagnostic 或 oracle 结果写成主 claim。
```
