# V7-HP-PAPER selector_v2.1 Status Report

Date: 2026-06-19 JST

## 1. Current Status

The `selector_v2.1_budgeted_risk_relax` 100-sample gate had passed on 2026-06-18, so the `final_1000` validation path was allowed.

As of this check, the following final_1000 stages have completed:

1. raw 1000-sample candidate reader evaluation;
2. v2.1 budgeted final selection over the raw candidate outputs;
3. per-example delta export;
4. bootstrap-style significance report generation.

No active `run_support_insertion_selector_v1.py` process remains for this run.

Main raw eval log:

`V7-HP-PAPER/selector_v2_1/logs/final_1000_raw_candidate_eval.nohup.log`

Final budget selection log:

`V7-HP-PAPER/selector_v2_1/logs/final_1000_budget_selection.log`

## 2. Completed final_1000 Files

Produced:

- `V7-HP-PAPER/selector_v2_1/outputs/final_1000/raw_candidate_eval/predictor_v2/candidate_rows.json`
- `V7-HP-PAPER/selector_v2_1/outputs/final_1000/raw_candidate_eval/predictor_v2/predictor_predictions.json`
- `V7-HP-PAPER/selector_v2_1/outputs/final_1000/raw_candidate_eval/selector_v1_100/selector_summary.json`
- `V7-HP-PAPER/selector_v2_1/outputs/final_1000/final_1000_summary.json`
- `V7-HP-PAPER/selector_v2_1/outputs/final_1000/per_example_delta.jsonl`
- `V7-HP-PAPER/selector_v2_1/outputs/final_1000/significance_report.json`

Note:

The directory name `selector_v1_100` inside `raw_candidate_eval` is inherited from the reusable v1 candidate-evaluation script. In this run it contains 1000-sample raw candidate outputs, not a 100-sample run.

## 3. 100-Sample Gate Recap

The 100-sample `selector_v2_1_best` result was:

| metric | value | delta |
| --- | ---: | ---: |
| answer_f1 | 0.6288 | +0.0133 |
| joint_f1 | 0.5121 | +0.0139 |
| support_recall@5 | 0.8050 | +0.0200 |
| sp_f1 | 0.7343 | +0.0286 |
| fallback_rate | 0.4000 | - |

100-sample gate status: passed.

This was the first PAPER selector variant to satisfy the intended operating point:

`answer protected + joint/support positive + fallback in budget range`.

## 4. Raw 1000 Candidate Eval

The raw candidate reader evaluation completed 6000 prompts:

- 1000 HotpotQA validation examples;
- 6 candidate contexts per example;
- reader: `google/flan-t5-large`;
- device: `cuda:0`.

The raw reusable selector output reported its own best selector as `without_predictor`, but it failed gate because fallback was `0.0`.

Raw best candidate-selector summary:

| metric | value |
| --- | ---: |
| n | 1000 |
| answer_access@5 | 0.8220 |
| support_recall@5 | 0.8155 |
| sp_f1 | 0.7429 |
| answer_em | 0.4750 |
| answer_f1 | 0.6066 |
| joint_f1 | 0.5155 |
| fallback_rate | 0.0000 |

Interpretation:

The raw candidate pool still contains support/joint signal, but unrestricted selection remains too aggressive and does not satisfy the abstention-aware policy.

## 5. final_1000 selector_v2.1 Result

The v2.1 final budget selection over the 1000 raw candidates selected:

| field | value |
| --- | --- |
| budget_select_count | 80 |
| available_actions | 566 |
| safe_answer_prob_threshold | 0.65 |
| support_gain_threshold | null |
| risk_penalty_weight | 0.0 |
| candidate_family | insert1_only |
| fallback_rate | 0.9200 |
| selected_count | 80 |

Final 1000 metrics:

| metric | value | delta |
| --- | ---: | ---: |
| answer_access@5 | 0.8330 | +0.0000 |
| support_recall@5 | 0.8190 | +0.0000 |
| sp_f1 | 0.7483 | +0.0000 |
| answer_em | 0.4800 | +0.0000 |
| answer_f1 | 0.6100 | +0.0000 |
| joint_f1 | 0.5170 | +0.0000 |

final_1000 gate status: failed.

Reason:

The final v2.1 selection preserved the raw baseline-equivalent metrics but did not create a positive delta on 1000. Fallback was within the broad allowed interval, but support/joint deltas were exactly zero.

## 6. Significance Report

Generated:

`V7-HP-PAPER/selector_v2_1/outputs/final_1000/significance_report.json`

Bootstrap summary:

| metric | mean_delta | 95% CI | p-value |
| --- | ---: | --- | ---: |
| answer_f1 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| joint_f1 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| support_recall@5 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| sp_f1 | 0.0000 | [0.0000, 0.0000] | 1.0000 |

Interpretation:

There is no statistically meaningful final_1000 gain for the current v2.1 selector implementation.

## 7. Diagnosis

The discrepancy is:

- 100-sample v2.1: clear positive signal;
- 1000 final v2.1: no delta.

Likely causes:

1. The 100-sample calibration overfit the small gate subset.
2. The final_1000 candidate scoring selected very few effective changed contexts.
3. The `insert1_only` family became too conservative at 1000 scale.
4. Many selected actions had no effective added/removed document difference after mapping into final candidate rows.
5. The final budget selected only 80/1000 actions, producing fallback `0.92`, far above the preferred 0.30-0.50 target.

The most important technical warning:

The 1000 run's `fallback_rate = 0.92` means the budgeted policy collapsed back toward baseline at scale. This is why all deltas became zero.

## 8. Paper-Level Interpretation

The result is still useful for the paper:

1. It confirms that 100-sample gate success alone is not enough.
2. It shows that abstention-aware selection can be too conservative when transferred to a broader validation split.
3. It strengthens the argument that budget calibration must be performed on a representative validation scale or via robust cross-validation, not only a small 100-sample gate.
4. The raw 1000 candidate pool still contains routing signal, but the final selector did not exploit it.

Current conclusion:

`selector_v2.1 proves the budgeted idea on the 100-sample gate, but does not yet scale to final_1000.`

## 9. Recommended Next Step

Do not claim final success from v2.1.

Recommended next action:

`selector_v2.2_scale_calibrated_budget`

Design:

1. Use the completed 1000 raw candidate eval as the candidate table.
2. Recalibrate budget directly on larger folds:
   - target selected_count around 300-500;
   - target fallback around 0.50-0.70 first, then tighten.
3. Avoid selecting candidates that produce no actual document change.
4. Add a hard `effective_context_changed = true` filter.
5. Use per-fold calibration rather than a single global top-B from 100-sample settings.
6. Re-run only the final selection logic, not the reader, because the 1000 raw candidate reader outputs already exist.

This is cheap compared with rerunning reader and is the right next move.

## 10. Current Decision

Current v2.1 final_1000 status:

`raw eval complete, final selection complete, no significant gain, not ready for paper main result`.

The best immediate path is to reuse the existing 1000 raw candidate table and improve only the scale-calibrated budget selector.
