# V7-HP-PAPER selector_v2.2 Scale-Calibrated Budget Report

## Purpose

v2.2 reuses the completed 1000 raw candidate reader outputs and performs query-level cross-fitted scale calibration. It filters ineffective actions and evaluates only held-out fold decisions.

## Main Result

- gate_pass: False
- answer_f1_delta: -0.0001
- joint_f1_delta: +0.0081
- support_recall_delta: +0.0075
- sp_f1_delta: +0.0103
- fallback_rate: 0.5000
- selected_effective_action_rate: 1.0000

## Audit

```json
{
  "total_queries": 1000,
  "total_candidate_actions": 5000,
  "available_actions": 5000,
  "effective_actions": 4517,
  "ineffective_actions": 483,
  "effective_action_rate": 0.9034,
  "selected_actions_in_v2_1": 80,
  "selected_effective_actions_in_v2_1": 295,
  "selected_ineffective_actions_in_v2_1": 105,
  "candidate_family_effective_rate": {
    "top4_bg1": 0.809,
    "insert1": 0.9655,
    "insert2": 0.994,
    "bridge": 0.783
  },
  "candidate_family_avg_delta_if_available": {
    "top4_bg1": {
      "answer_f1_delta": 0.0023625560112402228,
      "joint_f1_delta": 0.015635721337976978,
      "support_recall_delta": 0.015,
      "sp_f1_delta": 0.019714285714285712
    },
    "insert1": {
      "answer_f1_delta": -0.0017096168324429199,
      "joint_f1_delta": 0.008763928273245037,
      "support_recall_delta": 0.013,
      "sp_f1_delta": 0.016214285714285712
    },
    "insert2": {
      "answer_f1_delta": -0.005782034632034635,
      "joint_f1_delta": -0.014304617604617596,
      "support_recall_delta": -0.018,
      "sp_f1_delta": -0.024428571428571435
    },
    "bridge": {
      "answer_f1_delta": 0.0014542513042513049,
      "joint_f1_delta": 0.02857003552003553,
      "support_recall_delta": 0.03,
      "sp_f1_delta": 0.04114285714285716
    }
  }
}
```

## Significance

```json
{
  "n": 1000,
  "num_bootstrap_samples": 2000,
  "metrics": {
    "answer_f1": {
      "mean_delta": -7.236652236652241e-05,
      "ci95": [
        -0.010513924963924962,
        0.00993066378066378
      ],
      "p_value": 0.49
    },
    "joint_f1": {
      "mean_delta": 0.008121840857555143,
      "ci95": [
        -0.003768604411461552,
        0.020023809523809527
      ],
      "p_value": 0.0985
    },
    "support_recall@5": {
      "mean_delta": 0.0075,
      "ci95": [
        -0.0005,
        0.0155
      ],
      "p_value": 0.04
    },
    "sp_f1": {
      "mean_delta": 0.010285714285714283,
      "ci95": [
        -0.0004285714285714286,
        0.021285714285714283
      ],
      "p_value": 0.0315
    }
  }
}
```

## Diagnosis

```json
{
  "n_cases": 1000,
  "n_failure_cases": 971,
  "label_counts": {
    "candidate_pool_no_positive_action": 377,
    "wrong_action_selected": 423,
    "positive_action_rejected_by_budget": 119,
    "answer_gain_no_support_gain": 13,
    "under_abstention_answer_drop": 23,
    "support_gain_no_reader_gain": 12,
    "over_abstention": 4
  },
  "selected_but_answer_drop_count": 23,
  "selected_and_joint_gain_count": 41,
  "selected_and_support_gain_count": 41,
  "fallback_but_positive_action_exists_count": 119,
  "positive_action_rejected_count": 119,
  "ineffective_action_selected_count": 0,
  "candidate_pool_positive_rate": 0.223
}
```

## Oracle Gap Diagnostic

```json
{
  "queries_with_actions": 996,
  "oracle_best_joint_delta": 0.1411011809219546,
  "oracle_best_answer_safe_joint_delta": 0.1507981319915391,
  "oracle_support_delta": 0.09538152610441768,
  "positive_candidate_rate": 0.22389558232931728,
  "answer_safe_positive_candidate_rate": 0.22389558232931728,
  "selector_recall_of_positive_candidates": 0.18385650224215247,
  "diagnostic_only": true
}
```

## Interpretation

If gate passes, this is the first paper-ready no-leak selector result because thresholds are selected on train folds and evaluated on held-out queries. If it fails while the oracle gap is positive, the candidate space contains useful contexts but no-leak selection remains the bottleneck.