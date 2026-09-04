# V7-HP-PAPER selector_v2 Current Report

Date: 2026-06-17 JST

## 1. Purpose

`V7-HP-PAPER-selector-v2` continues from `support_insertion_selector_v1`, which improved support-side metrics and slightly improved `joint_f1`, but failed the 100-sample gate because `answer_f1` dropped below baseline.

The v2 hypothesis is:

> Support insertion should be a high-confidence event, not the default action. The selector should fallback to baseline unless no-leak support gain is clear, answer risk is low, and the safety predictor is confident.

The goal is not to maximize `support_recall@5` at all costs. The goal is safe useful insertion:

- preserve `answer_f1`;
- gain non-trivial support exposure;
- convert that support gain into positive `joint_f1`;
- maintain strict no-leak inference.

## 2. Implementation Status

New directory:

`/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/selector_v2/`

Implemented files:

- `build_v2_candidates.py`
- `train_safety_predictor_v3.py`
- `calibrate_v2_thresholds.py`
- `run_selector_v2_100.py`
- `run_selector_v2_ablation.py`
- `diagnose_selector_v2_failures.py`
- `run_selector_v2_100.sh`

The v2 run reuses v1's already generated 100-sample reader outcomes and no-leak candidate feature tables. This avoids rerunning 600 Flan-T5-Large prompts while still enforcing strict no-leak selector behavior at inference time.

Input reused from v1:

- `V7-HP-PAPER/outputs/predictor_v2/candidate_rows.json`
- `V7-HP-PAPER/outputs/predictor_v2/predictor_predictions.json`

## 3. Strict No-Leak Constraints

Inference-time selector does not use:

- gold supporting facts;
- gold supporting titles;
- gold answer string;
- answer string presence;
- current-query answer F1 / joint F1 / EM;
- current-query candidate outcome;
- oracle delta ranking.

Allowed signals:

- query and document text;
- retrieved titles;
- BM25 / dense / hybrid proxy scores;
- agent routing weight;
- context overlap and prefix stability;
- lexical/title/entity overlap;
- candidate-vs-baseline no-leak feature deltas;
- query-level split predictor probability from v1 outputs.

## 4. v2 Decision Design

The selector uses three gates:

1. Safety gate: `safe_answer_prob >= threshold`
2. Gain gate: `support_proxy_delta >= threshold`
3. Risk gate: `answer_risk_score <= threshold`

If no candidate survives, the selector falls back to baseline.

The calibrated v2 threshold configuration was:

```json
{
  "safe_answer_prob_threshold": 0.7,
  "support_gain_threshold": 0.0,
  "answer_risk_threshold": 0.3,
  "prefix_constraint": "keep_top2",
  "candidate_family": "insert1_only"
}
```

This setting was selected because it protects answer quality and keeps fallback in the desired conservative range.

## 5. Main Result

Baseline:

| metric | value |
| --- | ---: |
| answer_access@5 | 0.8100 |
| support_recall@5 | 0.7850 |
| sp_f1 | 0.7057 |
| answer_em | 0.4700 |
| answer_f1 | 0.6155 |
| joint_f1 | 0.4981 |

`selector_v2_full`:

| metric | value | delta |
| --- | ---: | ---: |
| answer_access@5 | 0.8100 | +0.0000 |
| support_recall@5 | 0.7850 | +0.0000 |
| sp_f1 | 0.7057 | +0.0000 |
| answer_em | 0.4700 | +0.0000 |
| answer_f1 | 0.6155 | +0.0000 |
| joint_f1 | 0.4981 | +0.0000 |

Selector behavior:

| statistic | value |
| --- | ---: |
| fallback_rate | 0.7200 |
| average_added_docs | 0.0200 |
| average_removed_docs | 0.0200 |
| prefix2_preserve_rate | 1.0000 |
| prefix3_preserve_rate | 0.9800 |
| safe_answer_prob_mean | 0.9500 |
| answer_risk_score_mean | 0.0218 |
| support_proxy_delta_mean | 0.0054 |

Selected candidate distribution:

| candidate | count |
| --- | ---: |
| baseline_fallback | 72 |
| keep_top3_bridge_insert1 | 19 |
| keep_top3_insert1_slot4 | 2 |
| keep_top2_insert1_slot3 | 6 |
| keep_top3_insert1_slot5 | 1 |

## 6. Gate Decision

Gate result: **failed**.

Reason:

- `answer_f1 >= baseline`: pass
- `fallback_rate = 0.72`: pass
- `support_recall@5 > baseline`: fail, delta `+0.0000`
- `sp_f1 >= baseline`: pass, but no gain
- `joint_f1 > baseline`: fail, delta `+0.0000`

Therefore, 1000 validation was not started.

## 7. Ablation Findings

| variant | answer_f1 | joint_f1 | support_recall@5 | sp_f1 | fallback | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| selector_v2_full | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | false |
| v2_no_gain_threshold | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | false |
| v2_no_predictor | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | false |
| v2_no_answer_risk_gate | 0.6258 | 0.5095 | 0.8050 | 0.7257 | 0.1900 | false |
| v2_insert1_only | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | false |
| v2_keep_top2 | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | false |
| v2_keep_top3 | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7400 | false |
| v2_fixed_threshold | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 1.0000 | false |
| v2_calibrated_threshold | 0.6155 | 0.4981 | 0.7850 | 0.7057 | 0.7200 | false |

Important signal:

`v2_no_answer_risk_gate` is the only variant that improves all reader metrics:

- `answer_f1`: `+0.0103`
- `joint_f1`: `+0.0114`
- `support_recall@5`: `+0.0200`
- `sp_f1`: `+0.0200`

But it fails the v2 policy constraint because fallback is only `0.19`, lower than the required `0.30`. This means a more aggressive insertion rule can work, but the current conservative threshold policy suppresses too many useful candidates.

## 8. Failure Diagnosis

Failure summary:

| label | count |
| --- | ---: |
| over_conservative_fallback | 72 |
| insufficient_support_gain | 28 |

Additional counters:

| counter | value |
| --- | ---: |
| selected_but_answer_drop_count | 0 |
| insert1_success_count | 0 |
| insert2_success_count | 0 |
| bridge_insert_success_count | 0 |

Interpretation:

v2 solved the answer-protection problem but became too abstention-heavy. It avoided answer damage, but the accepted candidates had too little support gain to move `support_recall@5` or `joint_f1`.

The key bottleneck has shifted:

- v1: too aggressive, answer damage;
- v2: too conservative, no useful gain;
- promising ablation: removing answer-risk gate improves all metrics but violates fallback constraint.

## 9. Paper Interpretation

The result supports the paper narrative:

Client-side routing exposes useful support candidates, but reader-facing context selection must be abstention-aware. However, excessive abstention erases the routing gain. The useful region appears to sit between v1's over-selection and v2's over-conservative gate.

Current status:

- routing signal: still positive;
- answer protection: achieved by v2;
- stable joint gain under conservative gate: not yet achieved;
- ready for 1000 validation: no.

## 10. Recommended Next Step

Do not start 1000 validation.

The next experiment should be `selector_v2.1` or `selector_v3_light`:

1. Keep the v2 default-fallback design.
2. Relax only the answer-risk gate, not the safety predictor.
3. Add a minimum/maximum insertion budget to force fallback into `0.30-0.50` rather than `0.19` or `0.72`.
4. Use the `v2_no_answer_risk_gate` setting as the upper candidate, then select only the top safest subset until fallback reaches the required interval.
5. Preserve prefix2, but allow prefix3 changes when support_proxy_delta and safe_answer_prob are both high.

This is the clearest path because `v2_no_answer_risk_gate` already shows the desired metric direction but needs abstention calibration.

## 11. Deliverables

Produced:

- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/selector_v2_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/per_example_delta.jsonl`
- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/failure_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/failure_cases.jsonl`
- `V7-HP-PAPER/selector_v2/outputs/ablation_100/ablation_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/threshold_calibration/threshold_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/predictor_v3/predictor_v3_summary.json`
- `V7-HP-PAPER/selector_v2/reports/v7_hp_paper_selector_v2_report.md`

Not produced:

- `final_1000` outputs, because the 100-sample gate failed.
