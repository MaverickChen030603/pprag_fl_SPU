# V7-HP-PAPER selector_v2 Report

## Purpose

`selector_v2` tests whether support insertion should be treated as a high-confidence event rather than a default action. It defaults to baseline and inserts only when safety, support gain, and answer-risk gates all pass.

## Gate Result

- gate_pass: False
- fallback_rate: 0.7200
- answer_f1_delta: -0.0000
- joint_f1_delta: -0.0000
- support_recall_delta: +0.0000
- sp_f1_delta: +0.0000

## Main Metrics

| mode | answer_f1 | joint_f1 | recall@5 | sp_f1 | d_answer | d_joint | d_recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.6155 | 0.4981 | 0.7850 | 0.7057 | +0.0000 | +0.0000 | +0.0000 |
| top4_bg1_balanced | 0.5627 | 0.4796 | 0.8150 | 0.7443 | -0.0529 | -0.0186 | +0.0300 |
| keep_top3_insert1_slot5 | 0.5763 | 0.5031 | 0.8200 | 0.7429 | -0.0392 | +0.0050 | +0.0350 |
| keep_top3_insert1_slot4 | 0.5763 | 0.5031 | 0.8200 | 0.7429 | -0.0392 | +0.0050 | +0.0350 |
| keep_top2_insert1_slot3 | 0.5740 | 0.4785 | 0.8050 | 0.7257 | -0.0415 | -0.0196 | +0.0200 |
| keep_top3_bridge_insert1 | 0.6080 | 0.5230 | 0.8350 | 0.7686 | -0.0075 | +0.0248 | +0.0500 |
| keep_top3_insert2_strict | 0.5723 | 0.4346 | 0.7400 | 0.6414 | -0.0432 | -0.0636 | -0.0450 |
| selector_v2_full | 0.6155 | 0.4981 | 0.7850 | 0.7057 | -0.0000 | -0.0000 | +0.0000 |

## Calibration

```json
{
  "chosen_thresholds": {
    "safe_answer_prob_threshold": 0.7,
    "support_gain_threshold": 0.0,
    "answer_risk_threshold": 0.3,
    "prefix_constraint": "keep_top2",
    "candidate_family": "insert1_only"
  },
  "best_calibration_metrics": {
    "n": 100,
    "answer_access_at_k": 0.81,
    "support_recall_at_k": 0.785,
    "sp_f1": 0.7057142857142855,
    "answer_em": 0.47,
    "answer_f1": 0.6155086580086578,
    "joint_f1": 0.49813512677798377,
    "variant": "selector_v2_full",
    "safe_answer_prob_threshold": 0.7,
    "support_gain_threshold": 0.0,
    "answer_risk_threshold": 0.3,
    "prefix_constraint": "keep_top2",
    "candidate_family": "insert1_only",
    "fallback_rate": 0.72,
    "average_added_docs": 0.02,
    "average_removed_docs": 0.02,
    "prefix2_preserve_rate": 1.0,
    "prefix3_preserve_rate": 0.98,
    "safe_answer_prob_mean": 0.95002501677658,
    "answer_risk_score_mean": 0.021785714285714287,
    "support_proxy_delta_mean": 0.005430540975812683,
    "selected_candidate_distribution": {
      "baseline_fallback": 72,
      "keep_top3_bridge_insert1": 19,
      "keep_top3_insert1_slot4": 2,
      "keep_top2_insert1_slot3": 6,
      "keep_top3_insert1_slot5": 1
    },
    "gate_pass": false,
    "delta_answer_f1": -1.1102230246251565e-16,
    "delta_joint_f1": -1.1102230246251565e-16,
    "delta_support_recall_at_k": 0.0
  },
  "grid_size": 1080,
  "feasible_count": 0,
  "top10": [
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.1,
      "prefix_constraint": "keep_top2",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.1,
      "prefix_constraint": "keep_top3",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.15,
      "prefix_constraint": "keep_top2",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.15,
      "prefix_constraint": "keep_top3",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.2,
      "prefix_constraint": "keep_top2",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.2,
      "prefix_constraint": "keep_top3",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.25,
      "prefix_constraint": "keep_top2",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.25,
      "prefix_constraint": "keep_top3",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.3,
      "prefix_constraint": "keep_top2",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.72,
      "average_added_docs": 0.02,
      "average_removed_docs": 0.02,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 0.98,
      "safe_answer_prob_mean": 0.95002501677658,
      "answer_risk_score_mean": 0.021785714285714287,
      "support_proxy_delta_mean": 0.005430540975812683,
      "selected_candidate_distribution": {
        "baseline_fallback": 72,
        "keep_top3_bridge_insert1": 19,
        "keep_top3_insert1_slot4": 2,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    },
    {
      "n": 100,
      "answer_access_at_k": 0.81,
      "support_recall_at_k": 0.785,
      "sp_f1": 0.7057142857142855,
      "answer_em": 0.47,
      "answer_f1": 0.6155086580086578,
      "joint_f1": 0.49813512677798377,
      "variant": "selector_v2_full",
      "safe_answer_prob_threshold": 0.7,
      "support_gain_threshold": 0.0,
      "answer_risk_threshold": 0.3,
      "prefix_constraint": "keep_top3",
      "candidate_family": "insert1_only",
      "fallback_rate": 0.74,
      "average_added_docs": 0.0,
      "average_removed_docs": 0.0,
      "prefix2_preserve_rate": 1.0,
      "prefix3_preserve_rate": 1.0,
      "safe_answer_prob_mean": 0.9513186157417539,
      "answer_risk_score_mean": 0.0003846153846153848,
      "support_proxy_delta_mean": 0.0,
      "selected_candidate_distribution": {
        "baseline_fallback": 74,
        "keep_top3_bridge_insert1": 19,
        "keep_top2_insert1_slot3": 6,
        "keep_top3_insert1_slot5": 1
      },
      "gate_pass": false,
      "delta_answer_f1": -1.1102230246251565e-16,
      "delta_joint_f1": -1.1102230246251565e-16,
      "delta_support_recall_at_k": 0.0
    }
  ]
}
```

## Ablation

| variant | answer_f1 | joint_f1 | recall@5 | sp_f1 | fallback | gate_pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| selector_v2_full | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | False |
| v2_no_gain_threshold | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | False |
| v2_no_predictor | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | False |
| v2_no_answer_risk_gate | 0.6258 | 0.5095 | 0.8050 | 0.7257 | 0.1900 | False |
| v2_insert1_only | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | False |
| v2_keep_top2 | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | False |
| v2_keep_top3 | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7400 | False |
| v2_fixed_threshold | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 1.0000 | False |
| v2_calibrated_threshold | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | False |

## Interpretation

v1 failed because it selected too many candidates with weak support gain. v2 raises abstention pressure through safety, gain, answer-risk, and prefix gates. If v2 still fails, the bottleneck is likely candidate pool quality or predictor calibration rather than another rerank sweep.