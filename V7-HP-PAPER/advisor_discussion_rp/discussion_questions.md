# Discussion Questions for Advisor

## 1. Problem Framing

1. 当前问题定位是否足够清楚：从 Federated RAG routing 转向 reader-safe context selection，这个转向是否合理？

2. `policy-action-to-reader gap` 是否适合作为论文的核心概念？这个名字是否易懂，还是需要换成更直观的说法，例如 `routing-to-reader gap`？

3. 论文主线应强调 Federated RAG，还是强调 Reader-safe Context Selection？如果强调后者，是否会削弱 Federated RAG 的新意？

## 2. Experimental Positioning

4. HP-hyper / BSP-DIAG 是否应放在 motivation / diagnostic section，而不是主结果？这种处理是否足够说服 reviewer？

5. HotpotQA v2.3 当前结果是否足够支撑投稿：joint_f1、support_recall@5、sp_f1 显著提升，但 answer_f1 只是 small non-significant positive delta。

6. 2Wiki 应该放在 limitation、appendix，还是 external diagnostic section？是否需要进一步弱化它，避免 reviewer 认为跨数据集泛化失败？

## 3. Claim Boundary

7. 论文是否可以把主 claim 写成：`significantly improves joint/support metrics while preserving answer_f1`？这个 claim 是否过强或过弱？

8. 是否应该避免在标题或摘要中使用过强的 Federated RAG claim，例如 “improving Federated RAG”，而改成 “action selection after federated routing”？

## 4. Additional Experiments

9. 是否必须补 multi-reader replication？如果资源有限，把 reader sensitivity 写入 limitation 是否足够？

10. 是否需要补充更多 case studies，展示 support gain 如何伤害 answer，以及 answer-neutral selector 如何避免这种错误？

## 5. Submission Strategy

11. 目标会议应考虑 EMNLP/NAACL Findings、COLING，还是尝试 main conference？

12. 如果投稿空间有限，HP-hyper / BSP-DIAG 应保留多少篇幅？是作为一节完整 diagnostic，还是压缩到 motivation + appendix？

## Suggested Meeting Goal

本次和导师讨论的目标不是确认所有技术细节，而是先确认三件事：

1. 论文主问题是否成立；
2. claim boundary 是否安全；
3. 下一步最值得补的实验是什么。

## Self-check

```text
ready_for_advisor_discussion: true
main_story_clear: true
technical_terms_explained: true
claim_boundary_respected: true
recommended_next_action: 开会时优先讨论 framing、claim 和补实验优先级，不要陷入所有实验细节。
```
