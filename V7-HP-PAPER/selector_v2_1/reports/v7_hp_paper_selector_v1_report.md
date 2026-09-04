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
| baseline | 1000 | 0.8330 | 0.8190 | 0.7483 | 0.4800 | 0.6100 | 0.5170 | +0.0000 | +0.0000 | +0.0000 |
| baseline_prefix_preserve_insert1 | 1000 | 0.8340 | 0.8300 | 0.7610 | 0.4790 | 0.6148 | 0.5298 | +0.0049 | +0.0128 | +0.0110 |
| baseline_prefix_preserve_insert2 | 1000 | 0.8200 | 0.8010 | 0.7239 | 0.4730 | 0.6042 | 0.5027 | -0.0058 | -0.0143 | -0.0180 |
| bridge_title_insert | 1000 | 0.8350 | 0.8490 | 0.7894 | 0.4800 | 0.6114 | 0.5456 | +0.0015 | +0.0286 | +0.0300 |
| support3_anchor2 | 1000 | 0.8300 | 0.8340 | 0.7680 | 0.4700 | 0.6016 | 0.5217 | -0.0083 | +0.0047 | +0.0150 |
| top4_bg1_balanced | 1000 | 0.8440 | 0.8340 | 0.7680 | 0.4820 | 0.6123 | 0.5326 | +0.0024 | +0.0156 | +0.0150 |

## Selector Ablation

| variant | answer_f1 | joint_f1 | recall@5 | sp_f1 | fallback | prefix2 | selected distribution | gate_pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| without_predictor | 0.6066 | 0.5155 | 0.8155 | 0.7429 | 0.0000 | 1.0000 | {"baseline_prefix_preserve_insert2": 411, "top4_bg1_balanced": 321, "support3_anchor2": 27, "bridge_title_insert": 173, "baseline_prefix_preserve_insert1": 68} | False |
| predictor_only | 0.6038 | 0.5132 | 0.8190 | 0.7483 | 0.4330 | 0.9880 | {"baseline_fallback": 433, "baseline_prefix_preserve_insert1": 28, "baseline_prefix_preserve_insert2": 32, "support3_anchor2": 28, "bridge_title_insert": 207, "top4_bg1_balanced": 272} | False |
| support_proxy_only | 0.6035 | 0.5253 | 0.8305 | 0.7626 | 0.0000 | 0.7390 | {"baseline_prefix_preserve_insert1": 377, "bridge_title_insert": 108, "top4_bg1_balanced": 247, "support3_anchor2": 241, "baseline_prefix_preserve_insert2": 27} | False |
| support_proxy_answer_risk | 0.6034 | 0.5083 | 0.8155 | 0.7437 | 0.4330 | 0.9880 | {"baseline_fallback": 433, "bridge_title_insert": 190, "baseline_prefix_preserve_insert2": 56, "support3_anchor2": 33, "top4_bg1_balanced": 257, "baseline_prefix_preserve_insert1": 31} | False |
| full_selector | 0.6020 | 0.5102 | 0.8180 | 0.7473 | 0.4330 | 0.9880 | {"baseline_fallback": 433, "bridge_title_insert": 202, "baseline_prefix_preserve_insert2": 58, "support3_anchor2": 29, "top4_bg1_balanced": 264, "baseline_prefix_preserve_insert1": 14} | False |

## Gate Decision

- best_selector: `without_predictor`
- answer_f1_delta: -0.0033
- joint_f1_delta: -0.0015
- support_recall_delta: -0.0035
- sp_f1_delta: -0.0054
- gate_pass: False

The 100-sample gate did not pass. Do not start 1000 validation; inspect failure_summary and improve predictor/selector calibration first.

## Failure Summary

```json
{
  "n_failure_cases": 955,
  "label_counts": {
    "insufficient_support_gain": 908,
    "predictor_false_unsafe": 12,
    "predictor_false_safe": 4,
    "context_replacement_loss": 28,
    "other": 3
  }
}
```