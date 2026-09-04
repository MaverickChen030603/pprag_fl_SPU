# V7-HP-PAPER selector_v2 Status Report

Date: 2026-06-18 JST

## 1. Server Status

Checked server project:

`/home/iiserver31/projects/FedE4RAG-main`

Current `V7-HP-PAPER/selector_v2` status:

- No active `selector_v2` / `V7-HP-PAPER` experiment process is running.
- The 100-sample selector_v2 gate check has completed.
- `final_1000` has not been launched.
- `final_1000_summary.json` does not exist, which is correct because the 100-sample gate failed.

Main selector_v2 log:

`V7-HP-PAPER/selector_v2/logs/selector_v2_100.log`

Main selector_v2 report:

`V7-HP-PAPER/selector_v2/reports/v7_hp_paper_selector_v2_report.md`

Latest current report:

`V7-HP-PAPER/selector_v2/reports/v7_hp_paper_selector_v2_current_report_latest.md`

## 2. Completed Outputs

The following required selector_v2 files are present:

- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/selector_v2_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/per_example_delta.jsonl`
- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/failure_cases.jsonl`
- `V7-HP-PAPER/selector_v2/outputs/selector_v2_100/failure_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/ablation_100/ablation_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/threshold_calibration/threshold_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/predictor_v3/predictor_v3_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/candidates/candidate_rows_v2_reuse.json`
- `V7-HP-PAPER/selector_v2/reports/v7_hp_paper_selector_v2_report.md`

The following files are intentionally absent:

- `V7-HP-PAPER/selector_v2/outputs/final_1000/final_1000_summary.json`
- `V7-HP-PAPER/selector_v2/outputs/final_1000/significance_report.json`
- `V7-HP-PAPER/selector_v2/reports/v7_hp_paper_selector_v2_final_1000_report.md`

Reason: selector_v2 did not pass the 100-sample gate, so 1000 validation must not start.

## 3. Experiment Purpose Recap

`selector_v2` was designed after v1 failed because `answer_f1` dropped below baseline.

The v2 goal was to verify whether a conservative, abstention-aware support insertion policy can:

1. preserve `answer_f1`;
2. retain or improve support exposure;
3. produce positive `joint_f1`;
4. keep strict no-leak inference;
5. avoid v1's overly aggressive candidate selection.

In short:

`v1` selected too often and hurt answer quality.

`v2` should fallback by default and insert only when safety, support gain, and answer-risk gates all pass.

## 4. Baseline Metrics

100-sample baseline:

| metric | value |
| --- | ---: |
| n | 100 |
| answer_access@5 | 0.8100 |
| support_recall@5 | 0.7850 |
| sp_f1 | 0.7057 |
| answer_em | 0.4700 |
| answer_f1 | 0.6155 |
| joint_f1 | 0.4981 |

## 5. selector_v2_full Metrics

`selector_v2_full`:

| metric | value | delta vs baseline |
| --- | ---: | ---: |
| answer_access@5 | 0.8100 | +0.0000 |
| support_recall@5 | 0.7850 | +0.0000 |
| sp_f1 | 0.7057 | +0.0000 |
| answer_em | 0.4700 | +0.0000 |
| answer_f1 | 0.6155 | +0.0000 |
| joint_f1 | 0.4981 | +0.0000 |

Gate status: **failed**.

Reason:

- `answer_f1 >= baseline`: passed
- `fallback_rate` in target conservative range: passed
- `support_recall@5 > baseline`: failed
- `joint_f1 > baseline`: failed

## 6. Selector Behavior

Calibrated thresholds:

| threshold | value |
| --- | --- |
| safe_answer_prob_threshold | 0.70 |
| support_gain_threshold | 0.00 |
| answer_risk_threshold | 0.30 |
| prefix_constraint | keep_top2 |
| candidate_family | insert1_only |

Behavior:

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

Selected distribution:

| candidate | count |
| --- | ---: |
| baseline_fallback | 72 |
| keep_top3_bridge_insert1 | 19 |
| keep_top3_insert1_slot4 | 2 |
| keep_top2_insert1_slot3 | 6 |
| keep_top3_insert1_slot5 | 1 |

## 7. Failure Diagnosis

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

The v2 selector successfully eliminated answer damage, but it became too conservative. Most examples fell back to baseline, and selected candidates had almost no support proxy margin. As a result, v2 preserved the baseline but did not create additional support or joint gain.

## 8. Important Ablation Signal

The most informative ablation remains:

`v2_no_answer_risk_gate`

It produced:

- `answer_f1 = 0.6258`, delta `+0.0103`
- `joint_f1 = 0.5095`, delta `+0.0114`
- `support_recall@5 = 0.8050`, delta `+0.0200`
- `sp_f1 = 0.7257`, delta `+0.0200`
- `fallback_rate = 0.1900`

It fails the current v2 policy constraint because fallback is below the required lower bound of `0.30`.

But this is the clearest positive signal in the selector_v2 family: when answer-risk gating is relaxed, the method can improve answer, support, and joint metrics simultaneously. The next task is to recover this positive behavior while raising fallback into the acceptable range.

## 9. Current Scientific Interpretation

The progression is now clear:

1. HP4 proved client-side routing can expose support evidence.
2. v1 proved predictor-based selection can retain support/joint signal but still damages answer.
3. v2 proved conservative fallback can protect answer, but over-abstention erases support gain.
4. The useful operating region likely lies between v1 and v2:
   - fallback should be higher than v1's `0.04`;
   - fallback should be lower than v2's `0.72`;
   - the promising target is roughly `0.30-0.50`.

This supports the paper narrative:

Reader-facing context selection must be abstention-aware, but not so conservative that routing gains never reach the reader.

## 10. Recommended Next Step

Do not run 1000 validation yet.

Recommended next experiment:

`selector_v2.1_budgeted_risk_relax`

Design:

1. Start from `v2_no_answer_risk_gate`, because it has the strongest metric signal.
2. Keep the safety predictor.
3. Keep prefix2 preservation.
4. Relax the answer-risk gate, but add a query budget:
   - allow only the top 50-70 candidate insertions by conservative score;
   - force fallback rate into `0.30-0.50`.
5. Use support gain and safety probability as the primary ranking keys.
6. Re-run 100-sample gate only.

Success condition:

- `answer_f1 >= baseline`
- `joint_f1 > baseline`
- `support_recall@5 > baseline`
- `sp_f1 >= baseline`
- `0.30 <= fallback_rate <= 0.95`

Only if this passes should `final_1000` be launched.

## 11. Current Decision

Current `selector_v2_full` is not ready for 1000 validation.

Status:

`answer protected, support gain suppressed, no 1000 launch`.
