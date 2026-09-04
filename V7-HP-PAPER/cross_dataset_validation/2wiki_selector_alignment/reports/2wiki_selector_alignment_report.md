# V7-HP-PAPER 2Wiki Selector Alignment Report

## 1. Purpose

This experiment upgrades the previous 2Wiki lexical/BM25 reader smoke into a selector-level validation attempt for `selector_v2.3_answer_neutral_positive_selector`.

## 2. What the Previous Smoke Proved

The previous 2Wiki dev-300 reader-backed smoke proved that the 2Wiki adapter and reader path work, and that lexical/BM25 routing is much stronger than raw context order. It did not prove HotpotQA v2.3 selector generalization.

## 3. Action Table and Feature Alignment

- queries: `300`
- actions: `2400`
- effective action rate: `0.7996`
- prefix2 preserve rate: `0.6275`
- prefix3 preserve rate: `0.6258`
- dense feature available: `False`
- safe answer mode: `heuristic_smoke_only`

Aligned feature set:

```text
support_proxy_delta
support_proxy_delta_vs_replaced_doc
answer_risk_score
title_bridge_score
prefix2_preserved
prefix3_preserved
num_added_docs
num_removed_docs
candidate_family
candidate_name
effective_context_changed
safe_answer_prob
```

## 4. Selector Smoke 300 Results

Main baseline is `bm25_or_lexical_routing`.

| method | answer_f1 | evidence_f1 | joint_f1 | answer_delta_vs_bm25 | evidence_delta_vs_bm25 | joint_delta_vs_bm25 | effective_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_order | 0.3652 | 0.3933 | 0.1824 | -0.0817 | -0.3337 | -0.1966 | 0.0000 |
| bm25_or_lexical_routing | 0.4469 | 0.7270 | 0.3790 | +0.0000 | +0.0000 | +0.0000 | 1.0000 |
| support_first_selector | 0.4452 | 0.7206 | 0.3741 | -0.0017 | -0.0064 | -0.0049 | 0.9967 |
| hotpot_v23_frozen_transfer | 0.4300 | 0.6614 | 0.3449 | -0.0170 | -0.0656 | -0.0341 | 0.9067 |
| 2wiki_v23_crossfit_selector | 0.3660 | 0.4003 | 0.1855 | -0.0809 | -0.3267 | -0.1935 | 0.0233 |
| no_safety_predictor | 0.3660 | 0.4027 | 0.1879 | -0.0809 | -0.3243 | -0.1911 | 0.0267 |
| no_support_features | 0.3652 | 0.3933 | 0.1824 | -0.0817 | -0.3337 | -0.1966 | 0.0000 |
| oracle_diagnostic_only | 0.6055 | 0.7898 | 0.5323 | +0.1586 | +0.0628 | +0.1533 | 0.9767 |

Gate:

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

## 5. No-Leak Audit

- audit status: `passed`
- query fold disjoint: `True`
- held-out outcome not used for inference: `True`
- oracle separated: `True`

## 6. Failure Diagnosis

```json
{
  "positive_action_available_but_not_selected": 142,
  "selector_underperforms_bm25": 55,
  "answer_drop_selected": 59,
  "support_positive_but_joint_negative": 12
}
```

## 7. 1000 Decision

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

## 8. Paper Recommendation

Recommendation: `pipeline_validation_only`

2Wiki results validate dataset transfer and lexical routing effectiveness, but do not yet establish selector-level generalization beyond a strong BM25 baseline.

## 9. MuSiQue Recommendation

Do not start MuSiQue as a broad stress test until the 2Wiki selector-level gap against BM25 is resolved or explicitly framed as a limitation. A MuSiQue run now would mostly test dataset plumbing rather than selector generalization.
