# V7-HP-PAPER 论文大纲

建议题目：

**Answer-Neutral Action Selection for Federated RAG Routing in Multi-Hop QA**

备选题目：

**Bridging Routing-Side Support Gains and Reader-Side Joint QA Gains under No-Leak Constraints**

## 摘要

要点：

1. Federated RAG 中，客户端 routing 可以发现有用 evidence，但下游 reader 对 context 组合非常敏感。
2. 直接插入 support-relevant context 可能提升 support metrics，却不一定提升 answer/joint QA。
3. 本文提出 answer-neutral positive-action selector，在 strict no-leak cross-fitting 下选择既保护 answer quality 又提升 joint utility 的 context action。
4. 在 HotpotQA 1000-query validation 上，v2.3 显著提升 joint_f1、support_recall@5、sp_f1，并保持 answer_f1 小幅正向但不显著。
5. 候选池分析显示 778/1000 queries 没有 paper-positive action，说明 candidate generation 仍是未来瓶颈。

## 1. Introduction

### 1.1 背景

- RAG 在 multi-hop QA 中依赖准确 evidence retrieval；
- Federated setting 下，客户端有局部语料/参数/路由信号；
- FL routing 能帮助暴露 rare/support evidence，但 aggregation/retrieval/reader pipeline 会压平或扭曲这些信号。

### 1.2 核心问题

Support-side gain 与 answer-side quality 不一致：

- routing 可能找到更多 support evidence；
- reader 输入被替换后，答案表述线索可能丢失；
- support_f1 提升不必然带来 joint_f1 或 answer_f1 提升。

### 1.3 本文贡献

建议写三点：

1. 提出 policy-action-to-reader gap：federated routing 的有效 action 需要 reader-aware selection 才能转化为 QA 收益。
2. 提出 answer-neutral positive-action selector，在 no-leak cross-fit 下选择 answer-safe 且 joint-beneficial 的 action。
3. 提供完整诊断：主结果、ablation、candidate pool quality、feature importance、failure analysis，说明收益来源与瓶颈。

## 2. Related Work

### 2.1 Retrieval-Augmented Generation for Multi-Hop QA

讨论 HotpotQA、多跳 evidence retrieval、reader sensitivity、supporting fact evaluation。

### 2.2 Federated Retrieval / Federated RAG

讨论 federated client-side knowledge、隐私约束、aggregation、routing 与 retrieval。

### 2.3 Reader-Aware Reranking and Context Selection

讨论 context reranking、answer-aware selection、避免 retrieval-only proxy 与 reader outcome 脱节。

### 2.4 No-Leak Evaluation and Cross-Fitting

强调不能用 held-out query outcome、gold answer、gold support 做 inference-time selection。

## 3. Problem Formulation

### 3.1 Federated RAG Action Selection

定义：

- query `q`;
- baseline context `C_b`;
- candidate action `a`;
- selected context `C_a`;
- reader output metrics：answer_f1、support_recall、sp_f1、joint_f1。

目标：

在不使用 held-out labels 的情况下选择 action，使：

```text
joint_f1_delta > 0
support_recall_delta > 0
sp_f1_delta >= 0
answer_f1_delta >= 0
```

### 3.2 Answer-Neutral Positive Action

定义 paper-positive action：

```text
answer_f1_delta >= 0
joint_f1_delta > 0
(support_recall_delta > 0 or sp_f1_delta >= 0)
```

### 3.3 No-Leak Constraint

Inference-time 禁止使用：

- gold answer；
- gold supporting facts/titles；
- current query reader outcome；
- oracle delta；
- held-out query label。

允许使用：

- query/document lexical features；
- support proxy；
- routing weight；
- title bridge score；
- safety predictor；
- cross-fitted model prediction。

## 4. Method

### 4.1 Candidate Generation from Federated Routing

简述 HP4/v2.x 如何产生 candidate actions：

- baseline context；
- insert/bridge/top-k variants；
- effective action filtering；
- support/routing features。

### 4.2 Answer-Neutral Positive Selector

核心方法：

- train folds 上构建 action labels；
- 训练 two-stage classifier / pairwise ranker；
- train folds 上校准 threshold 与 selected_fraction；
- held-out folds 上只用预测分数选择 action。

最终 cross-fit 使用：

- two-stage + all_effective_conservative：2 folds；
- pairwise_ranker + all_effective_conservative：3 folds；
- selected_fraction = 0.5。

### 4.3 Training Labels

列出：

- answer_safe；
- joint_positive；
- answer_safe_joint_positive；
- support_positive；
- paper_positive；
- answer_drop。

### 4.4 Features

包括：

- safe_answer_prob；
- support_proxy_delta；
- support_proxy_delta_vs_replaced_doc；
- agent_weight_delta；
- title_bridge_score；
- answer_risk_score；
- displacement_score；
- prefix preservation；
- candidate_family / candidate_name。

### 4.5 Cross-Fit Protocol

5-fold query-level split：

1. train selector on train queries；
2. calibrate threshold/budget on train queries；
3. apply to held-out queries；
4. aggregate held-out predictions。

强调 formal selector 与 oracle diagnostic 分离。

## 5. Experimental Setup

### 5.1 Dataset

HotpotQA validation subset，1000 queries。

### 5.2 Metrics

- answer_access@5；
- support_recall@5；
- sp_f1；
- answer_em；
- answer_f1；
- joint_f1；
- paired bootstrap significance。

### 5.3 Baselines and Variants

包括：

- baseline；
- v2.2 scale-calibrated budget；
- v2.3 main；
- two_stage；
- paper_positive_classifier；
- answer_drop_rejector_support_ranker；
- constrained_regression；
- no answer-neutral constraint；
- no support features；
- no safety predictor；
- oracle diagnostic upper bound。

### 5.4 No-Leak Audit

简述审计结论：

- disjoint query folds；
- train-only calibration；
- no held-out outcome inference；
- oracle separated；
- no gold answer/support inference features。

## 6. Main Results

主表放置：

| Method | answer_f1_delta | joint_f1_delta | support_recall_delta | sp_f1_delta | gate |
|---|---:|---:|---:|---:|---|
| v2.2 | -0.0001 | +0.0081 | +0.0075 | +0.0103 | false |
| v2.3 | +0.0023 | +0.0150 | +0.0190 | +0.0254 | true |

正文重点：

- v2.3 首次同时满足 answer preservation 与 joint/support gain；
- joint_f1 显著：p = 0.0245；
- support_recall@5 与 sp_f1 显著；
- answer_f1 为小幅正向但不显著。

## 7. Ablation Study

建议组织：

### 7.1 Selector Architecture

- two_stage 有稳定正信号；
- pairwise mixed configuration 最强；
- paper_positive_classifier 较弱但通过 gate。

### 7.2 Answer-Neutral Constraint

- no_safety_predictor 导致 answer_f1_delta = -0.0070，说明 safety/answer-neutral 信号必要。

### 7.3 Support/Routing Features

- no_support_features 仍有正信号但弱于主结果，说明 support/routing 特征贡献有效。

### 7.4 Answer-Drop Rejector

- answer_drop_rejector_support_ranker 失败，说明单纯拒绝 answer drop 不足以提升 positive recall。

## 8. Candidate Pool and Feature Diagnostics

### 8.1 Candidate Pool Quality

关键结论：

- paper_positive_rate = 0.0896；
- 778/1000 queries 没有 paper-positive action；
- insert1 family 的 paper_positive_rate 最高，为 0.1075。

这说明 candidate generation 是主要上限。

### 8.2 Feature Importance

重点解释：

- routing/support features：agent_weight_delta、support_proxy_delta、title_bridge_score 有正区分度；
- safety predictor：不是简单越高越好，而是帮助 answer-neutral 控制；
- answer_risk_score：用于风险解释，不用于声称 answer 显著提升。

## 9. Case Studies

三类 case：

1. Success cases：answer-safe + joint/support gain；
2. Answer-neutral cases：answer_f1 基本不变但 joint/support 提升；
3. Failure cases：candidate pool 无正例、positive action 未选中、wrong action、answer drop。

注意写明 gold answer 只用于展示分析，不作为 selector feature。

## 10. Failure Analysis

主失败类别：

| Failure | Count |
|---|---:|
| candidate_pool_no_positive_action | 778 |
| positive_action_available_but_not_selected | 102 |
| wrong_action_selected | 41 |
| answer_drop_selected | 2 |
| support_positive_but_joint_negative | 4 |

解释：

- 最大瓶颈是没有可选正例；
- selector 仍有 ranking 改进空间；
- answer-neutral constraint 有效，因为 answer_drop_selected 很少；
- support improvement 不总是 reader improvement。

## 11. Discussion

### 11.1 What Worked

- answer-neutral selection 将 support gain 转化为 joint_f1 gain；
- strict no-leak cross-fit 提升可信度；
- positive candidate recall 从 v2.2 的 0.1839 提升到 v2.3 的 0.3288。

### 11.2 What Did Not Fully Work

- answer_f1 没有显著提升；
- candidate pool 正例稀疏；
- oracle upper bound 仍远高于 formal selector；
- 并非所有 multi-hop query 都可受益。

### 11.3 Implications for Federated RAG

Federated routing 不应只用 retrieval proxy 评估，还需要 reader-aware、answer-neutral action selection。

## 12. Limitations

建议写：

1. 当前实验基于 HotpotQA 1000-query subset；
2. answer_f1 正向但不显著；
3. candidate generation 是瓶颈；
4. no-leak audit 是 artifact/source-level；
5. oracle diagnostic 只说明潜力，不是正式方法。

## 13. Conclusion

结论段建议：

本文证明，在 Federated RAG 多跳问答中，routing-side support gains 可以通过 answer-neutral positive-action selector 转化为 downstream joint_f1 gains。v2.3 在 strict no-leak query-level cross-fitting 下显著提升 joint_f1、support_recall@5 与 sp_f1，并保持 answer_f1 小幅正向。实验同时揭示 candidate pool quality 是后续主要瓶颈。

## 附录建议

Appendix A：No-leak audit details  
Appendix B：Full ablation table  
Appendix C：Feature importance table  
Appendix D：Case studies  
Appendix E：Candidate pool quality breakdown  
Appendix F：Claim boundary memo
