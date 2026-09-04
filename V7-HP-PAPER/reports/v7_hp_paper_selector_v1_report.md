# V7-HP-PAPER support_insertion_selector_v1 Report

## Experiment Purpose

V7-HP-PAPER targets the HP4 Phase 3 bottleneck: routing improves support exposure, but aggressive context replacement can lower answer_f1. The selector therefore chooses no-leak support insertions only when a query-level split reader-safety predictor and explicit answer-risk features consider them safe.

## Strict No-Leak Design

- Inference features exclude gold support titles, gold answer strings, answer presence, and current-query reader outcomes.
- Reader outcome labels are used only for training/evaluating predictor_v2 with leave-one-query-out splits.
- Candidate selection uses query/document lexical, dense/BM25 proxy, agent weights, context stability, entity/title overlap, and predictor probability.

## 100-Sample Metrics

| mode | n | access@5 | recall@5 | sp_f1 | answer_em | answer_f1 | joint_f1 | d_answer | d_joint | d_recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100 | 0.8100 | 0.7850 | 0.7057 | 0.4700 | 0.6155 | 0.4981 | +0.0000 | +0.0000 | +0.0000 |
| baseline_prefix_preserve_insert1 | 100 | 0.8000 | 0.8200 | 0.7429 | 0.4200 | 0.5763 | 0.5031 | -0.0392 | +0.0050 | +0.0350 |
| baseline_prefix_preserve_insert2 | 100 | 0.7700 | 0.7400 | 0.6414 | 0.4300 | 0.5723 | 0.4346 | -0.0432 | -0.0636 | -0.0450 |
| bridge_title_insert | 100 | 0.8100 | 0.8350 | 0.7686 | 0.4700 | 0.6080 | 0.5230 | -0.0075 | +0.0248 | +0.0500 |
| support3_anchor2 | 100 | 0.8100 | 0.8050 | 0.7257 | 0.4500 | 0.5740 | 0.4785 | -0.0415 | -0.0196 | +0.0200 |
| top4_bg1_balanced | 100 | 0.8100 | 0.8150 | 0.7443 | 0.4300 | 0.5627 | 0.4796 | -0.0529 | -0.0186 | +0.0300 |

## Selector Ablation

| variant | answer_f1 | joint_f1 | recall@5 | sp_f1 | fallback | prefix2 | selected distribution | gate_pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| without_predictor | 0.5925 | 0.4938 | 0.7850 | 0.7057 | 0.0000 | 1.0000 | {"top4_bg1_balanced": 31, "bridge_title_insert": 15, "baseline_prefix_preserve_insert2": 38, "baseline_prefix_preserve_insert1": 11, "support3_anchor2": 5} | False |
| predictor_only | 0.6025 | 0.5052 | 0.8050 | 0.7300 | 0.0400 | 0.9100 | {"top4_bg1_balanced": 38, "bridge_title_insert": 35, "baseline_fallback": 4, "baseline_prefix_preserve_insert2": 5, "support3_anchor2": 9, "baseline_prefix_preserve_insert1": 9} | False |
| support_proxy_only | 0.5547 | 0.4725 | 0.8150 | 0.7357 | 0.0000 | 0.7100 | {"baseline_prefix_preserve_insert1": 39, "support3_anchor2": 20, "top4_bg1_balanced": 29, "bridge_title_insert": 10, "baseline_prefix_preserve_insert2": 2} | False |
| support_proxy_answer_risk | 0.5985 | 0.4855 | 0.7850 | 0.7057 | 0.0400 | 0.9700 | {"top4_bg1_balanced": 27, "baseline_prefix_preserve_insert1": 26, "baseline_prefix_preserve_insert2": 21, "bridge_title_insert": 15, "baseline_fallback": 4, "support3_anchor2": 7} | False |
| full_selector | 0.6025 | 0.4895 | 0.7800 | 0.6986 | 0.0400 | 0.9700 | {"top4_bg1_balanced": 33, "bridge_title_insert": 21, "baseline_prefix_preserve_insert2": 20, "baseline_fallback": 4, "baseline_prefix_preserve_insert1": 15, "support3_anchor2": 7} | False |

## Gate Decision

- best_selector: `predictor_only`
- answer_f1_delta: -0.0130
- joint_f1_delta: +0.0071
- support_recall_delta: +0.0200
- sp_f1_delta: +0.0243
- gate_pass: False

The 100-sample gate did not pass. Do not start 1000 validation; inspect failure_summary and improve predictor/selector calibration first.

## Failure Summary

```json
{
  "n_failure_cases": 94,
  "label_counts": {
    "insufficient_support_gain": 87,
    "baseline_already_optimal": 4,
    "predictor_false_safe": 1,
    "context_replacement_loss": 2
  }
}
```