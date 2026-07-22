# V16 投稿与继续实验建议

**最终状态：** `analysis_paper_only`  
**机器判定：** `hold_or_redirect`  
**建议：** 停止通用 composer 训练，转向 composition opportunity 分析或新的 candidate-pool/direct-set-utility protocol。

## 1. 是否存在真实 composition synergy？

存在，但当前证据只到 Oracle 层。Exact100 的六个 dataset-reader 单元中，mean StrictSynJoint 为 +0.0305 至 +0.0656，paired-bootstrap 95% CI 均排除 0。它证明 best composed context 严格超过同一 query 的全部合法 single edits。

## 2. Composition-only positive query 占比多少？

- 2Wiki：FLAN 6%，UnifiedQA 5%。
- HotpotQA：FLAN 2%，UnifiedQA 5%。
- MuSiQue：FLAN 11%，UnifiedQA 10%。

只有 MuSiQue 在两个 reader 上达到预注册的 10% 门槛。

## 3. Learned composer 能实现多少 Oracle synergy？

未知，且没有进行实验。Checkpoint 1 未通过，预注册禁止进入 composer 训练。不能把 Oracle 上界写成 learned realization。

## 4. 是否显著优于 best single edit？

Oracle best composed 显著优于 best single edit，六个 Joint/composite CIs 均高于 0。Learned policy 没有训练，因此不存在 learned-composer 显著性结论。

## 5. 是否显著优于旧 Full？

未测试。V16 Phase-A 是 Oracle feasibility study，不是与 V14 Full 的 end-to-end matched comparison。不得从 Oracle delta 推断优于旧 Full。

## 6. 是否缩小或逆转与 CrossEncoder 的 Joint 差距？

未测试。CrossEncoder 仅作为固定 ordering/feature 合同的一部分，不构成 V16 learned method baseline。当前不能声称缩小、逆转或改善该差距。

## 7. 是否在 HotpotQA、2Wiki、MuSiQue 重复？

Strict Oracle synergy 在三个数据集均重复。Composition-only >=10% 的机会密度只在 MuSiQue 重复。因此“协同存在”可跨数据集表述，“composition 是主要修复机制”不可跨数据集表述。

## 8. 是否跨 reader 成立？

Oracle StrictSynJoint 在 FLAN-T5-Large 与 UnifiedQA-T5-Large 上方向一致且显著。Answer StrictSyn 在 HotpotQA/MuSiQue 跨 reader 成立，在 2Wiki 两个 reader 上均不显著。

## 9. Top-20 是否提高 composition opportunity？

没有运行。Checkpoint 1 已基于冻结 Top-10 失败，事后用 Top-20 救门槛会改变预注册 protocol。未来只能在新的独立 protocol 中检验。

## 10. 收益是否仅来自更多 candidate 或计算？

Oracle 对照共享同一 Top-10 pool、同一五文档 budget、同一 reader，并相对全部合法 single edits 定义 StrictSyn，因此结果不是简单的 pool 或 context-budget 扩张。但 exhaustive composition 搜索确实评估更多 contexts；由于没有 learned/compute-matched inference，本阶段不能证明部署收益不依赖额外搜索计算。

## 11. Action composition 是否可作为主要创新？

目前不可以作为通用方法创新。它可作为分析对象和严格测量框架：固定 K 状态依赖 edits、best-single matched Oracle、composition-only opportunity。只有未来 learned policy 在至少两个数据集上实现显著收益后，才可升级为方法主张。

## 12. 最大剩余弱点是什么？

最大弱点是 learnability 与 opportunity density 均未建立。Oracle 协同的 median 为 0，composition-only rate 在两个数据集仅 2%-6%；同时没有 learned composer、推理延迟、harm、capacity-control 或 final-test 证据。

## 13. 适合 ECIR 还是 ARR？

更适合 ECIR 分析论文，而非 ARR 强方法论文。可聚焦候选池缺失、single-action opportunity、composition-only opportunity、reader sensitivity 和 strict synergy 的跨数据集差异。若仅以当前结果投 ARR，核心方法审稿人会合理追问 learned realization 与强基线。

## 14. 是否应继续实验或终止该方向？

应终止当前 protocol 下的 composer 训练，不应继续追逐门槛。保留两条新路线：

1. **分析路线：** 扩大独立样本，研究 MuSiQue 上 composition-only opportunity 与 hop/pool/evidence dispersion 的关系。
2. **方法重启路线：** 先提升真实 candidate-pool opportunity 或训练 direct set utility；在全新预注册 split 上重新执行 Oracle Checkpoint，再决定是否训练 composer。

## 总体建议

当前最诚实且有研究价值的定位是：Oracle composition synergy 是稳定现象，但其独占机会在真实 Top-10 pool 中稀疏且依赖数据集。项目应收束为“何时组合动作值得做”的分析，而不是继续包装为尚未被 learned policy 实现的通用方法突破。
