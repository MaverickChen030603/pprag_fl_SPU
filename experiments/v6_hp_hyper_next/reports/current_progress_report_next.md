# V6-HP-hyper 下一阶段当前进展

- 报告时间：2026-06-18 19:03 JST
- 本地仓库：`/Users/iilab/PPRAG_FL/FedE4RAG-main/FedE4RAG-main`
- 服务器仓库：`/home/iiserver31/projects/FedE4RAG-main`
- 实验目录：`experiments/v6_hp_hyper_next/`

## 1. 已完成部分

Task A、subset smoke test、Task B1 `hotpot_all_1000 / seed=42` 已完成。本轮新增完成了 B1 的 method identity audit，并启动了 Task B2 `hotpot_hard_1000 / seed=42`。

## 2. 资源检查

服务器检查时间：

```text
Thu Jun 18 18:58 JST 2026
```

资源状态：

| Item | Status |
| --- | --- |
| `/home` | 90% used, 380G available |
| `experiments/v6_hp_hyper_next` | 1.4G |
| GPU0 | available before B2 launch |
| GPU1/GPU2 | idle |
| GPU3 | 3596MiB used by unrelated process |
| system load | high, around 54-56 |

结论：磁盘低于 95% 风险线，可以继续小规模实验；但 CPU load 很高，因此 B2 运行速度明显慢于理想情况。

## 3. Task B2 状态

B2 已启动：

```text
subset = hotpot_hard_1000
seed = 42
methods = V3, V4, V5, V6-HP-hyper anchor
target_payload = 0.070134 ± 0.002
```

运行信息：

```text
PID = 95454
log = experiments/v6_hp_hyper_next/logs/same_payload_b2_hard1000_20260618_185840.nohup.log
raw = experiments/v6_hp_hyper_next/results/same_payload_b2_hard1000_raw.csv
```

当前进度：

```text
B2 正在运行 V3_topk2_fixed。
日志已确认 cuda:0 初始化成功，进入 FLGO 训练。
截至检查时，尚未产生 B2 raw CSV。
```

因此，B3 `hotpot_hard_500` 尚未启动。按照任务规则，B3 需要等 B2 成功、payload 与 subset 读取确认无误后再运行。

## 4. Method Identity Audit 结果

已对 B1 `hotpot_all_1000 / seed=42` 的真实服务器 run_dir 执行 method identity audit。

产物：

```text
experiments/v6_hp_hyper_next/results/method_identity_audit_raw.jsonl
experiments/v6_hp_hyper_next/results/method_identity_audit_summary.csv
experiments/v6_hp_hyper_next/reports/method_identity_audit_report.md
```

关键结果：

| Method | unique block sets | top blocks | utility memory | hard weighting |
| --- | ---: | --- | --- | --- |
| V3_topk2_fixed | 4 | `pooler:50; encoder.layer.8:45; encoder.layer.6:2; encoder.layer.7:2; encoder.layer.2:1` | not recorded | not recorded |
| V4_topk2_fixed | 4 | `pooler:50; encoder.layer.8:46; encoder.layer.9:2; encoder.layer.2:1; encoder.layer.1:1` | False | True |
| V5_topk2_fixed | 4 | same as V4 | False | True |
| V6_HP_hyper_anchor | 4 | same as V4/V5 | False | True |

Pairwise identity:

| Pair | identical selection |
| --- | --- |
| V3 vs V4 | False |
| V3 vs V5 | False |
| V3 vs V6 | False |
| V4 vs V5 | True |
| V4 vs V6 | True |
| V5 vs V6 | True |

解释：

```text
V4/V5/V6 same-payload topk2 fixed configurations collapse to equivalent block-selection behavior under the current implementation on all_1000 / seed=42.
```

这解释了为什么 B1 中 V4/V5/V6 的 downstream 指标几乎完全一致。V3 的 selection 与它们不完全相同，但最终 downstream metrics 仍高度接近，说明 all_1000 仍存在明显评测不敏感问题。

限制：

```text
当前 round_logs 没有保存 block_score_mean / block_score_std / top_selected_score_mean / top_selected_score_std，因此 audit 只能验证 selected blocks/layers 是否一致，不能验证 score distribution 是否一致。
```

## 5. 当前判断

已确认：

```text
1. all_1000 / seed=42 下，同 payload 对齐成功。
2. V6 在 all_1000 上没有明显优于 V3/V4/V5。
3. V4/V5/V6 在 all_1000 / seed=42 下 selection behavior 完全一致。
4. B2 hard_1000 已启动，是下一步决定性测试。
```

尚未确认：

```text
1. V6 是否能在 hard_1000 上超过 baseline。
2. hard split 是否能拉开方法差异。
3. 是否值得扩展 seeds 43/44。
4. 是否应进入 V6 ablation。
```

## 6. 下一步规则

等待 B2 完成后：

1. 如果 V6 在 `hard_1000` 上 MRR 或 F1 相对最佳 baseline 提升达到约 `>=0.01`，再考虑扩 seeds 或跑 B3。
2. 如果 V6 与 baseline 差距 `<0.005`，暂停 baseline 多 seed，转向 V6 ablation。
3. 如果 B2 中 V4/V5/V6 仍 selection 完全一致且结果高度相同，应输出 `bug_or_equivalence_diagnosis.md`，检查方法路径是否收敛到等价实现。
4. adaptive same-payload verification 暂缓，直到 hard split 证明方法差异足够敏感。

## 7. 当前最强可支持 claim

```text
Under strict same-payload constraints, V6-HP-hyper preserves FedRAG retrieval performance at payload≈0.0701. However, based on all_1000 / seed=42 and current method identity audit, V6 does not yet show a practically large advantage over V4/V5 because their topk2 fixed selection behavior is currently identical. The decisive evidence should come from hard_1000/hard_500 evaluation and method-level selection behavior analysis.
```
