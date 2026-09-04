# V7-HP-PAPER 2Wiki Selector Alignment Current Report

## 1. 当前状态

服务器当前没有 `2wiki_selector_alignment` 相关进程在运行。本轮 selector alignment smoke 已完成，1000 formal validation 因 gate 未通过而按规则跳过。

已完成：

- 2Wiki action table 300
- v2.3-compatible feature table 300
- dev-300 selector smoke with reader outcomes
- ablation summary
- failure diagnostics
- no-leak audit
- gated 1000 skip decision
- selector alignment report and paper recommendation

## 2. Action Table 与 Feature Alignment

- num_queries: `300`
- num_actions: `2400`
- actions_per_query: `8.0`
- effective_action_rate: `0.7996`
- avg_added_docs: `1.3638`
- avg_removed_docs: `1.3725`
- prefix2_preserve_rate: `0.6275`
- prefix3_preserve_rate: `0.6258`
- dense_feature_available: `False`
- safe_answer_prob_mode: `heuristic_smoke_only`

说明：dense feature 当前不可用；`safe_answer_prob` 在 smoke 阶段是 heuristic safety proxy，不应被写成正式训练出的 safety predictor。

## 3. Selector Smoke 300 结果

主基准是 `bm25_or_lexical_routing`，不是 raw context order。

| method | answer_f1 | evidence_f1 | joint_f1 | answer_delta_vs_bm25 | evidence_delta_vs_bm25 | joint_delta_vs_bm25 | effective_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_order | 0.3652 | 0.3933 | 0.1824 | -0.0817 | -0.3337 | -0.1966 | 0.0000 |
| bm25_or_lexical_routing | 0.4469 | 0.7270 | 0.3790 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| support_first_selector | 0.4452 | 0.7206 | 0.3741 | -0.0017 | -0.0064 | -0.0049 | 0.9967 |
| hotpot_v23_frozen_transfer | 0.4300 | 0.6614 | 0.3449 | -0.0170 | -0.0656 | -0.0341 | 0.9067 |
| 2wiki_v23_crossfit_selector | 0.3660 | 0.4003 | 0.1855 | -0.0809 | -0.3267 | -0.1935 | 0.0233 |
| no_safety_predictor | 0.3660 | 0.4027 | 0.1879 | -0.0809 | -0.3243 | -0.1911 | 0.0267 |
| no_support_features | 0.3652 | 0.3933 | 0.1824 | -0.0817 | -0.3337 | -0.1966 | 0.0000 |
| oracle_diagnostic_only | 0.6055 | 0.7898 | 0.5323 | 0.1586 | 0.0628 | 0.1533 | 0.9767 |

## 4. Gate 结论

```json
{
  "passed": false,
  "answer_f1_delta_vs_bm25": -0.08094025146966323,
  "evidence_recall_delta_vs_bm25": -0.2941666666666667,
  "evidence_f1_delta_vs_bm25": -0.3266931216931213,
  "joint_f1_delta_vs_bm25": -0.19346548611674658,
  "selected_effective_action_rate": 0.023333333333333334,
  "decision": "stop_at_smoke_300"
}
```

Gate 未通过。关键原因是 `2wiki_v23_crossfit_selector` 相对 BM25 strong baseline 在 answer、evidence、joint 三类指标上全部为负，且 `selected_effective_action_rate` 只有 `0.0233`。

因此不运行 1000 reader validation；这符合实验说明中的 stop rule。

## 5. 失败诊断

```json
{
  "positive_action_available_but_not_selected": 142,
  "selector_underperforms_bm25": 55,
  "answer_drop_selected": 59,
  "support_positive_but_joint_negative": 12
}
```

主要失败不是数据管线，而是 selector 在 2Wiki 上偏向 fallback / 非有效 action，无法稳定选择 BM25 已经找到的高质量上下文。

## 6. No-Leak Audit

- audit status: `passed`
- query_fold_disjoint: `True`
- held_out_outcome_not_used_for_inference: `True`
- gold_answer_support_not_used_as_inference_feature: `True`
- oracle_separated_from_formal_method: `True`

## 7. 1000 状态

```json
{
  "status": "skipped_gate_not_passed",
  "reason": "2Wiki selector smoke 300 did not satisfy BM25-relative gate; formal 1000 reader validation is intentionally not run.",
  "smoke_gate": {
    "passed": false,
    "answer_f1_delta_vs_bm25": -0.08094025146966323,
    "evidence_recall_delta_vs_bm25": -0.2941666666666667,
    "evidence_f1_delta_vs_bm25": -0.3266931216931213,
    "joint_f1_delta_vs_bm25": -0.19346548611674658,
    "selected_effective_action_rate": 0.023333333333333334,
    "decision": "stop_at_smoke_300"
  },
  "limitation": "2Wiki pipeline works, but v2.3 selector does not yet improve over a strong BM25/lexical baseline."
}
```

## 8. 当前论文口径

可以写：

> 2Wiki results validate dataset transfer and lexical routing effectiveness, but do not yet establish selector-level generalization beyond a strong BM25 baseline.

不能写：

> HotpotQA v2.3 selector generalizes to 2WikiMultiHopQA.

## 9. 下一步建议

不要直接启动 MuSiQue。当前瓶颈是 2Wiki action-feature alignment / selector decision rule 未超过 BM25，而不是缺另一个数据集。若继续推进，应优先修复 2Wiki action candidate generation 与 safety calibration，再考虑 1000 或 MuSiQue。

## 10. 关键文件

```text
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/action_table_300/action_table_summary.json
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/action_table_300/feature_summary.json
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/selector_smoke_300/summary.json
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/selector_smoke_300/significance_report.json
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/selector_smoke_300/failure_summary.json
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/audit/2wiki_no_leak_audit.md
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/reports/2wiki_selector_alignment_report.md
V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/reports/2wiki_paper_recommendation.md
```
