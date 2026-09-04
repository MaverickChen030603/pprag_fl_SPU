# V7-HP-PAPER selector_v2.3 answer-neutral positive selector 报告

## 1. 实验目的

v2.3 旨在解决 v2.2 的核心瓶颈：support-side 指标已有正信号，但 answer_f1 微弱为负，且 positive candidate recall 只有 0.1839。v2.3 不重跑 reader，复用 v2.2 effective action table，在 query-level cross-fitting 下训练 answer-neutral positive-action selector。

## 2. 标签分布

```json
{
  "num_actions": 5000,
  "num_queries": 1000,
  "answer_safe_rate": 0.9468,
  "joint_positive_rate": 0.0914,
  "answer_safe_joint_positive_rate": 0.0908,
  "paper_positive_rate": 0.0896,
  "answer_drop_rate": 0.0532,
  "large_answer_drop_rate": 0.053,
  "positive_actions_per_query_distribution": {
    "0": 778,
    "1": 91,
    "2": 69,
    "3": 33,
    "4": 25,
    "5": 4
  },
  "queries_with_no_positive_action": 778
}
```

## 3. Final 1000 结果

- `answer_f1_delta` = `0.002259235209235211`
- `joint_f1_delta` = `0.015012780869923525`
- `support_recall_delta` = `0.019000000000000017`
- `sp_f1_delta` = `0.025428571428571467`
- `fallback_rate` = `0.5`
- `positive_candidate_recall` = `0.32882882882882886`
- `selected_answer_drop_rate` = `0.058`
- `gate_pass` = `True`
- `paper_main_recommended` = `True`

## 4. Significance

```json
{
  "n": 1000,
  "num_bootstrap_samples": 2000,
  "metrics": {
    "answer_f1": {
      "mean_delta": 0.0022592352092352095,
      "ci95": [
        -0.011415367965367965,
        0.01576847041847042
      ],
      "p_value": 0.3625
    },
    "joint_f1": {
      "mean_delta": 0.015012780869923725,
      "ci95": [
        0.00010612244897959131,
        0.030237590187590185
      ],
      "p_value": 0.0245
    },
    "support_recall@5": {
      "mean_delta": 0.019,
      "ci95": [
        0.0085,
        0.0295
      ],
      "p_value": 0.0
    },
    "sp_f1": {
      "mean_delta": 0.025428571428571425,
      "ci95": [
        0.01057142857142857,
        0.039285714285714285
      ],
      "p_value": 0.0
    }
  }
}
```

## 5. Ablation 摘要

- `ablation_two_stage`: answer `0.005117476967476997`, joint `0.011273298923298869`, support `0.01200000000000001`, sp `0.014999999999999902`, recall `0.2927927927927928`
- `ablation_paper_positive_classifier`: answer `0.0017174769674770385`, joint `0.007587584637584577`, support `0.01200000000000001`, sp `0.014999999999999902`, recall `0.28378378378378377`
- `ablation_answer_drop_rejector_support_ranker`: answer `-0.0005393615156771281`, joint `0.007811137191964135`, support `0.012500000000000067`, sp `0.01657142857142846`, recall `0.12162162162162163`
- `ablation_constrained_regression`: answer `0.005001603951603983`, joint `0.009174721310435507`, support `0.010500000000000065`, sp `0.015000000000000013`, recall `0.2882882882882883`
- `ablation_no_answer_constraint`: answer `0.00125923520923521`, joint `0.01325195835910109`, support `0.018500000000000072`, sp `0.0247142857142858`, recall `0.31981981981981983`
- `ablation_no_support_features`: answer `0.00017252747252760603`, joint `0.012707508364650955`, support `0.013500000000000068`, sp `0.018428571428571794`, recall `0.3063063063063063`
- `ablation_no_safety_predictor`: answer `-0.006959812409812249`, joint `0.003985291692434445`, support `0.01100000000000001`, sp `0.013571428571428568`, recall `0.26576576576576577`
- `ablation_all_effective`: answer `0.005117476967476997`, joint `0.011273298923298869`, support `0.01200000000000001`, sp `0.014999999999999902`, recall `0.2927927927927928`
- `ablation_insert1_plus_bridge`: answer `0.005117476967476997`, joint `0.011273298923298869`, support `0.01200000000000001`, sp `0.014999999999999902`, recall `0.2927927927927928`
- `ablation_v2_2_support_first`: answer `-7.23665223665293e-05`, joint `0.008121840857555074`, support `0.007500000000000062`, sp `0.01028571428571412`, recall `None`

## 6. Failure Diagnosis

```json
{
  "n_cases": 1000,
  "label_counts": {
    "candidate_pool_no_positive_action": 778,
    "selected_positive": 73,
    "positive_action_available_but_not_selected": 102,
    "wrong_action_selected": 41,
    "answer_drop_selected": 2,
    "support_positive_but_joint_negative": 4
  },
  "positive_candidate_recall": 0.32882882882882886,
  "answer_safe_positive_candidate_recall": 0.3273542600896861,
  "selected_answer_drop_count": 29,
  "selected_answer_drop_rate": 0.058,
  "positive_action_available_but_not_selected_count": 102,
  "wrong_action_selected_count": 41,
  "model_false_positive_count": 43,
  "model_false_negative_count": 102,
  "candidate_pool_no_positive_action_count": 778,
  "candidate_pool_no_answer_safe_positive_count": 0
}
```

## 7. 论文判断

v2.3 通过 strict no-leak gate，可作为论文主结果候选。核心叙事是 federated routing 暴露有用 support candidates，answer-neutral selector 将其转化为 downstream gains。
