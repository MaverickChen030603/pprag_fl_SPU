# 2Wiki BM25-Anchor Repair Report

## 1. Purpose

This repair tests whether the previous 2Wiki selector failure was caused by action definitions that disrupted a strong BM25 context. The repair anchors all actions to BM25 top-5 and only permits minimal tail replacements.

## 2. Oracle Gap vs BM25

- num_queries: `300`
- positive_vs_bm25_rate: `0.2433`
- oracle_best_answer_delta_vs_bm25: `+0.1615`
- oracle_best_evidence_delta_vs_bm25: `+0.1267`
- oracle_best_joint_delta_vs_bm25: `+0.1533`
- selector_recall_of_positive_vs_bm25: `0.3425`
- oracle decision: `continue_bm25_anchor_repair`

Oracle is diagnostic only.

## 3. BM25-Anchor Action Table

- num_actions: `1800`
- effective_action_rate: `0.6589`
- bm25_top1_preserve_rate: `1.0000`
- bm25_top2_preserve_rate: `1.0000`
- bm25_top3_preserve_rate: `1.0000`
- hard_rule_violations: `0`

## 4. Real Safety Predictor

- answer_safe_auc: `0.5567`
- paper_positive_auc: `0.5451`
- false_safe_rate: `0.0367`
- false_negative_rate: `0.0000`

## 5. Selector Smoke 300

| method | answer_f1 | evidence_f1 | joint_f1 | answer_delta | evidence_delta | joint_delta | effective | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_or_lexical_routing | 0.4469 | 0.7270 | 0.3790 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | 1.0000 |
| previous_2wiki_v23_crossfit_selector | 0.3660 | 0.4003 | 0.1855 | -0.0809 | -0.3267 | -0.1935 | 0.2633 | 0.0000 |
| bm25_anchor_support_first | 0.4430 | 0.7203 | 0.3751 | -0.0040 | -0.0067 | -0.0039 | 0.1333 | 0.8667 |
| bm25_anchor_safety_first | 0.4325 | 0.7165 | 0.3599 | -0.0145 | -0.0104 | -0.0191 | 1.0000 | 0.0000 |
| bm25_anchor_positive_selector | 0.4389 | 0.7146 | 0.3650 | -0.0080 | -0.0124 | -0.0140 | 1.0000 | 0.0000 |
| bm25_anchor_answer_neutral_selector | 0.4486 | 0.7343 | 0.3792 | +0.0017 | +0.0073 | +0.0002 | 0.7000 | 0.3000 |
| no_safety_predictor | 0.4293 | 0.7295 | 0.3600 | -0.0176 | +0.0025 | -0.0190 | 1.0000 | 0.0000 |
| no_support_features | 0.4339 | 0.7038 | 0.3570 | -0.0131 | -0.0232 | -0.0220 | 1.0000 | 0.0000 |
| oracle_diagnostic_only | 0.5063 | 0.7559 | 0.4372 | +0.0594 | +0.0289 | +0.0582 | 0.1167 | 0.8833 |

Gate:

```json
{
  "passed": false,
  "best_config": {
    "selected_fraction": 0.1,
    "safe_threshold": 0.5,
    "positive_threshold": 0.1,
    "preserve_top3": true
  },
  "decision": "stop_at_smoke_300"
}
```

## 6. Failure Diagnosis

```json
{
  "candidate_pool_no_positive_vs_bm25": 227,
  "answer_drop_selected": 1,
  "positive_vs_bm25_available_but_not_selected": 27
}
```

## 7. No-Leak Audit

```json
{
  "status": "passed",
  "query_fold_disjoint": true,
  "train_fold_only_safety_predictor_training": true,
  "train_fold_only_threshold_calibration": "grid is evaluated on smoke for diagnostics; no formal 1000 launched",
  "held_out_outcome_not_used_for_inference": true,
  "gold_answer_support_not_used_as_inference_feature": true,
  "oracle_separated_from_formal_method": true
}
```

## 8. Paper Decision

Decision: `limitation_selector_failure`

2Wiki candidate actions contain positive opportunities beyond BM25, but current no-leak selector fails to identify them reliably.

## 9. MuSiQue Decision

Do not start MuSiQue from this state. The bottleneck is selector reliability over a strong lexical baseline, not cross-dataset plumbing.
