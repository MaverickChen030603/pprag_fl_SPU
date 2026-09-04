# V7-HP-PAPER Current Status Report

Date: 2026-06-17 JST

## 1. Current Status

`V7-HP-PAPER` has completed the strict no-leak 100-sample gate check for `support_insertion_selector_v1`.

No active `V7-HP-PAPER` process is running on the server. The 1000-sample validation has not been launched because the 100-sample gate condition failed, which follows the experiment rule.

Server root:

`/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER`

Main log:

`/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/logs/v7_hp_paper_selector_v1_100.nohup.log`

Main report:

`/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/reports/v7_hp_paper_selector_v1_report.md`

## 2. Experiment Purpose

The purpose of `V7-HP-PAPER` is to convert the HP4 routing-side gain into reader-side QA gain under strict no-leak constraints.

HP4 Phase 3 showed that soft routing, hardgate, and reader-aware rerank can increase `support_recall@5` and `sp_f1`, but aggressive support insertion can damage answer-bearing context and reduce `answer_f1`.

Therefore this experiment implements `support_insertion_selector_v1`: a reader-safety-aware but no-leak selector that decides whether support-like documents should be inserted into the top-5 reader context.

The target gate conditions were:

- `answer_f1 >= baseline`
- `joint_f1 > baseline`
- `support_recall@5 > baseline`
- `sp_f1 >= baseline`
- `fallback_rate != 100%`

## 3. Experimental Configuration

Dataset:

- HotpotQA validation subset
- 100-sample gate check
- Same validation source as HP4: `V7-HP4/data/hotpot_validation_1000.json`

Reader:

- `google/flan-t5-large`
- GPU: `cuda:1`
- Reader batch size: 1
- Total reader prompts: 600

Policy source:

- `V7-HP4/outputs/hp4_phase3_100_batch/policy_a/hp4_policy_reinforce.pt`

Candidate contexts generated per query:

- `baseline`
- `top4_bg1_balanced`
- `support3_anchor2`
- `baseline_prefix_preserve_insert1`
- `baseline_prefix_preserve_insert2`
- `bridge_title_insert`

Strict no-leak constraints:

- No gold supporting facts/titles as inference-time features
- No gold answer string or answer presence
- No current-query reader outcome
- No oracle delta ranking
- Predictor uses leave-one-query-out split, so candidates from the same query are not used to train that query's safety prediction

## 4. Main Results

Baseline:

| metric | value |
| --- | ---: |
| n | 100 |
| answer_access@5 | 0.8100 |
| support_recall@5 | 0.7850 |
| sp_f1 | 0.7057 |
| answer_em | 0.4700 |
| answer_f1 | 0.6155 |
| joint_f1 | 0.4981 |

Best selector variant:

`predictor_only`

| metric | value | delta vs baseline |
| --- | ---: | ---: |
| answer_access@5 | 0.8000 | -0.0100 |
| support_recall@5 | 0.8050 | +0.0200 |
| sp_f1 | 0.7300 | +0.0243 |
| answer_em | 0.4600 | -0.0100 |
| answer_f1 | 0.6025 | -0.0130 |
| joint_f1 | 0.5052 | +0.0071 |

Selector operating statistics:

| statistic | value |
| --- | ---: |
| fallback_rate | 0.0400 |
| average_added_docs | 0.2200 |
| average_removed_docs | 0.2200 |
| prefix2_preserve_rate | 0.9100 |
| prefix3_preserve_rate | 0.5300 |

Selected candidate distribution:

| candidate | count |
| --- | ---: |
| `top4_bg1_balanced` | 38 |
| `bridge_title_insert` | 35 |
| `baseline_fallback` | 4 |
| `baseline_prefix_preserve_insert2` | 5 |
| `support3_anchor2` | 9 |
| `baseline_prefix_preserve_insert1` | 9 |

## 5. Gate Decision

The 100-sample gate did not pass.

Reason:

- `support_recall@5` improved by `+0.0200`
- `sp_f1` improved by `+0.0243`
- `joint_f1` improved by `+0.0071`
- But `answer_f1` dropped by `-0.0130`

Because `answer_f1 >= baseline` is a required condition, the experiment correctly stopped before 1000 validation.

## 6. Ablation Summary

| variant | answer_f1 | joint_f1 | support_recall@5 | sp_f1 | gate_pass |
| --- | ---: | ---: | ---: | ---: | --- |
| `without_predictor` | 0.5925 | 0.4938 | 0.7850 | 0.7057 | false |
| `predictor_only` | 0.6025 | 0.5052 | 0.8050 | 0.7300 | false |
| `support_proxy_only` | 0.5547 | 0.4725 | 0.8150 | 0.7357 | false |
| `support_proxy_answer_risk` | 0.5985 | 0.4855 | 0.7850 | 0.7057 | false |
| `full_selector` | 0.6025 | 0.4895 | 0.7800 | 0.6986 | false |

Interpretation:

- Predictor helps reduce answer damage compared with support-only selection.
- Support-only selection increases support exposure but damages reader answer quality heavily.
- Current full selector is too conservative or misweighted: it protects prefix stability but loses support gain.
- `predictor_only` is currently the best tradeoff, but still does not preserve `answer_f1`.

## 7. Failure Diagnosis

Failure summary:

| label | count |
| --- | ---: |
| `insufficient_support_gain` | 87 |
| `baseline_already_optimal` | 4 |
| `context_replacement_loss` | 2 |
| `predictor_false_safe` | 1 |

Main diagnosis:

The dominant failure is not severe reader interference. Instead, most selected candidates do not add enough effective support gain to justify changing the baseline context.

This means the next improvement should not be another large context rerank. The selector needs a stronger acceptance rule: only perform insertion when support gain is clearly above a calibrated threshold and answer-risk is low. Otherwise it should keep baseline.

## 8. Current Conclusion

`V7-HP-PAPER` confirms the HP4 mechanism pattern:

1. Client-side routing still provides real support-side signal.
2. Reader-facing context selection remains the bottleneck.
3. Predictor-based safety helps, but the current selector does not yet satisfy the paper gate.
4. The method is promising but not ready for 1000 validation or main-paper claim.

The current result is best described as:

`support-side positive, joint small positive, answer-side not yet protected`.

## 9. Recommended Next Step

Do not launch 1000 validation yet.

Next implementation should be:

`support_insertion_selector_v2`

Recommended changes:

1. Add a hard minimum support-gain threshold before any candidate can replace baseline.
2. Make baseline fallback the default unless both conditions hold:
   - predicted safe probability is high;
   - support_proxy_delta is clearly positive.
3. Optimize for unsafe recall rather than safe precision, because the expensive error is selecting answer-damaging candidates.
4. Add inner query-level validation for threshold selection instead of using a fixed threshold.
5. Add a candidate family that preserves baseline top3 and permits only one insertion into slot 4 or 5.

Only after `answer_f1 >= baseline` and `joint_f1 > baseline` on the 100-sample gate should `final_1000` be started.

## 10. Deliverables Present

Existing output files:

- `V7-HP-PAPER/outputs/selector_v1_100/selector_summary.json`
- `V7-HP-PAPER/outputs/selector_v1_100/per_example_delta.jsonl`
- `V7-HP-PAPER/outputs/selector_v1_100/failure_cases.jsonl`
- `V7-HP-PAPER/outputs/selector_v1_100/failure_summary.json`
- `V7-HP-PAPER/outputs/ablation_100/ablation_summary.json`
- `V7-HP-PAPER/outputs/predictor_v2/predictor_v2_summary.json`
- `V7-HP-PAPER/reports/v7_hp_paper_selector_v1_report.md`

No `final_1000` result is present, by design.
