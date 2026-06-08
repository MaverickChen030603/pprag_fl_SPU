# V7-HP2 实验方案

V7-HP2 不直接宣称 agent 成功，而是针对 V7-HP1 的主要问题继续优化：HP1 在 official fullwiki supporting-fact 上有弱正信号，但没有转化为 T5-small reader 的真实 answer EM/F1 收益。

核心假设：如果 agent reward 只看 retrieval proxy，可能选择到对检索排序有益、但 reader 不容易利用的 block。HP2 将 reader feedback proxy 混入 agent memory，使 agent 更偏向选择能提升 evidence readability / answer generation 的参数块。

## Reader-Aware Reward

训练时不在每个 round 直接跑完整 T5 reader，避免成本失控。HP2 使用轻量 reader feedback proxy：

```text
utility_hp2(block) = (1 - w) * retrieval_proxy(block) + w * reader_feedback_proxy(block)
```

其中 `reader_feedback_proxy` 由三部分组成：

- reader-sensitive block prior：更重视 BERT upper encoder layers 与 pooler；
- evidence sharpness：由 block delta 的 max/mean/variance 估计；
- client hardness：hard client 上适度放大 reader feedback。

默认 `reader_feedback_weight=0.35`，`reader_feedback_scale=1.25`。

## 评估

- Official fullwiki supporting-fact/joint：HotpotQA dev 分层 300 条；
- Reader/generator QA：同一 dev300，同一 top retrieved context，`google-t5/t5-small` 生成 answer，计算 Hotpot normalization EM/F1；
- 判断标准：reader-aware agent 必须至少在 official supporting-fact 上优于 baseline，并进一步观察 reader EM/F1 是否提升。
