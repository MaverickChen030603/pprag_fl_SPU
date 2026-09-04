# V7-HP3 结果解读、归档与论文分析

生成日期: 2026-06-10  
实验目录: `/home/iiserver31/projects/FedE4RAG-main/V7-HP3`  
正式报告: `实验分析报告/V7-HP3/v7_hp3_reset_hard_reader_latest.md`

## 1. 实验定位

V7-HP3 是在 V7-HP2 打平之后启动的 reset 实验，目的不是继续堆规模，而是检验三个更激进的假设是否能把 agent 与 baseline 拉开：

1. 将 reader 从弱 T5-small 升级到更强的 `google/flan-t5-large`，让下游 QA 对检索质量更敏感。
2. 将 reader-aware reward 改为高对比阶梯奖励：高质量 block 给 +10，低质量 block 给 -5，中间置 0。
3. 从 V7-HP2 per-query 成绩单反筛 Recoverable-Hard 100，去除过易题与绝望题，保留更可能体现策略差异的 hard-case 子集。

这一轮实验因此是一个“敏感性压力测试”：如果 agent memory / rarity / reader feedback 真的改变了上传选择，并且这种改变能传导到检索与生成，HP3 应该比 HP2 更容易出现正向信号。

## 2. 完成状态

- 上游训练: 12/12 完成。
- Official HotpotQA fullwiki-context eval: 12/12 完成。
- Strong-reader eval: 12/12 完成。
- Reader 模型: `google/flan-t5-large`。
- 样本: Recoverable-Hard 100。
- 硬错误: 无 traceback、OOM、killed、runtime error；仅出现 FLGO visualize 警告，不影响训练或评估。

## 3. Recoverable-Hard 100 诊断

Hard100 的筛选结果本身非常关键：

| 指标 | 数值 |
| --- | ---: |
| 目标样本数 | 100 |
| 实际样本数 | 100 |
| 严格 recoverable 数 | 1 |
| 选入严格 recoverable 数 | 1 |
| fallback medium-hard 候选数 | 178 |
| easy removed | 7 |
| impossible removed | 46 |

解释：严格 recoverable 只有 1 条，说明 HP2 的 per-query 成绩单已经高度同质化。也就是说，baseline 错、agent 能明显救回来的题几乎不存在；Hard100 主要由 medium-hard fallback 组成，而不是理想的“agent 可恢复难题”。这为 HP3 的后续打平埋下了结构性原因。

## 4. Official 评估结果

| 方法 | profile | runs | answer_access@k | sp_f1 | joint_f1 |
| --- | --- | ---: | ---: | ---: | ---: |
| agent_memory_v7hp2 | hp2_reader_memory_agent | 3 | 0.2200 | 0.3374 | 0.0021 |
| agent_tail_v7hp2 | hp2_reader_tail_agent | 3 | 0.2200 | 0.3397 | 0.0021 |
| hypernet_v6 | hp2_baseline_adaptive_v6 | 3 | 0.2200 | 0.3374 | 0.0021 |
| hypernet_v6 | hp2_baseline_hypernet_v6 | 3 | 0.2200 | 0.3374 | 0.0021 |

Official agent-baseline best joint_f1 gap: `+0.0000`。

唯一可见的微小差异是 `agent_tail_v7hp2` 的 `sp_f1` 从 0.3374 到 0.3397，约 +0.0023；但 answer access 和 joint_f1 完全打平，因此不能作为有效正信号。

## 5. Strong Reader 评估结果

| 方法 | profile | runs | answer_em | answer_f1 | joint_f1 | answer_access@k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| agent_memory_v7hp2 | hp2_reader_memory_agent | 3 | 0.4900 | 0.5180 | 0.1656 | 0.2200 |
| agent_tail_v7hp2 | hp2_reader_tail_agent | 3 | 0.4900 | 0.5180 | 0.1656 | 0.2200 |
| hypernet_v6 | hp2_baseline_adaptive_v6 | 3 | 0.4900 | 0.5180 | 0.1656 | 0.2200 |
| hypernet_v6 | hp2_baseline_hypernet_v6 | 3 | 0.4900 | 0.5180 | 0.1656 | 0.2200 |

Reader agent-baseline best joint_f1 gap: `+0.0000`。

强 reader 没有放大差异，反而说明 reader 在所有方法拿到的 retrieved context 上看到的是几乎同一批可用证据；换言之，瓶颈不在 reader 能力不足，而在检索/上传策略没有实质改变最终上下文。

## 6. 结论判断

HP3 不能宣称 V7 agent 成功。更准确的论文表述应为：

> 在 HotpotQA Recoverable-Hard 100 上，即使引入强 reader 与高对比 reader-aware reward，agent memory/tail 策略仍未在 official joint_f1 或 reader joint_f1 上超过 baseline。这表明当前 agent 的 block selection 差异没有有效传导到最终 retrieved context 与 answer generation。

这是一个清晰的负结果，但不是无价值结果。它排除了三个先前假设：

1. 不是单纯因为 reader 太弱导致看不到差距。
2. 不是单纯因为 reward 太平滑导致看不到差距。
3. 不是单纯因为样本太容易导致看不到差距。

真正的问题更可能是：当前 agent 改变的是上传 budget/utility 的局部选择，但这些选择在 FedAvg/embedding 更新/检索排序链路中被抹平，最终 top-k retrieved context 基本不变。

## 7. 论文分析价值

HP3 可以写入论文的“诊断实验”或“边界分析”章节，作用是解释方法为何需要进一步改造，而不是作为主结果。

推荐论文叙事：

- 主线不要说 HP3 证明 agent 有效。
- HP3 作为 negative diagnostic：说明 proxy-level agent selection 不足以保证 downstream QA improvement。
- 将问题归纳为“policy-action-to-context gap”：agent policy 改变上传选择，但没有足够强地改变最终检索上下文。
- 下一阶段方法必须把 reward 从离线 proxy 升级到在线 context-level / reader-level credit assignment。

可写入论文的关键句：

> The HP3 reset study shows that strengthening the reader and sharpening the reward are insufficient when the selected parameter blocks do not induce measurable changes in retrieved evidence. This suggests that the bottleneck lies in credit assignment from downstream QA outcomes back to upload decisions, rather than in reader capacity alone.

## 8. 归档清单

建议归档以下材料作为 V7-HP3 证据包：

- `V7-HP3/run_v7_hp3_reset_full_pipeline.sh`
- `V7-HP3/prepare_recoverable_hard100.py`
- `V7-HP3/write_hp3_reset_analysis.py`
- `V7-HP3/data/hotpot_recoverable_hard100.meta.json`
- `V7-HP3/outputs/hotpot_official_fullwiki_hard100/official_eval_all_summary.json`
- `V7-HP3/outputs/hotpot_reader_strong_hard100/reader_eval_all_summary.json`
- `V7-HP3/logs/v7_hp3_reset_full_pipeline.nohup.log`
- `实验分析报告/V7-HP3/v7_hp3_reset_hard_reader_latest.md`
- 本文件: `V7_HP3_result_interpretation_archive_paper_analysis_cn.md`

## 9. 下一阶段优化方向

HP3 之后不建议继续只调 reader 或 reward 常数。更有效的下一步是开一个 V7-HP4 或 V8 分支，聚焦让 agent action 真实影响 retrieved context：

1. 做 context-delta audit：逐 query 比较各方法 top-k 文档/段落是否不同。如果 top-k overlap 过高，说明训练阶段策略差异没有落到检索空间。
2. 做 oracle rerank 上限：在 Hard100 上用 supporting facts 做 oracle rerank，估计当前 retriever pool 是否包含正确证据。如果 pool 都没有证据，agent selection 再强也无从改善 reader。
3. 将 reward 改为 query-level online reward：每轮或每若干轮真正跑小批量 retrieval + reader，按 answer_f1 / support_f1 回传到客户端和 block，而不是只用离线 proxy。
4. 加入 diversity/anti-collapse 约束：惩罚不同策略产生相同 top-k 或相同 uploaded block pattern，强制 agent 探索不同证据路径。
5. 缩小任务为可恢复 micro-benchmark：先用 20-50 条人工确认“baseline top-k 缺证据、agent 有机会通过少量更新改变 top-k”的问题，验证机制成立后再扩展。

## 10. 最终归档结论

V7-HP3 已完成并可归档。该实验没有给出 agent 正向指标，但给出了明确的失败模式：强 reader 与高对比 reward 无法弥补 policy selection 到 retrieved context 的传导断裂。论文上应将其作为诊断证据，支撑下一阶段提出更直接的 context-level credit assignment 和 online reader reward。
