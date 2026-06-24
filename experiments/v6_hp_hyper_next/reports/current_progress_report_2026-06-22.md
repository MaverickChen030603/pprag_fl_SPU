# V6-HP-hyper Current Progress Report

Date: 2026-06-22 JST

## 1. Server Progress Check

The queued S1 scorelog anchor job successfully waited for a safe GPU window and completed on the server.

Server status at check time:

- Server time: 2026-06-22 02:06 JST.
- Current GPUs are again occupied by other workloads; no V6-HP-hyper job is currently running.
- The previous queued scorelog job finished at 2026-06-21 00:42:43 JST.
- No selection-diversity ablation job has been launched yet.
- No B3 `hotpot_hard_500` or multi-seed job has been launched.

Evidence:

- Queue log: `experiments/v6_hp_hyper_next/logs/scorelog_anchor_hard1000_queue_20260619_152032.log`
- Anchor raw CSV: `experiments/v6_hp_hyper_next/results/scorelog_anchor_hard1000_raw.csv`
- Score raw JSONL: `experiments/v6_hp_hyper_next/results/score_logging_raw.jsonl`
- Score summary CSV: `experiments/v6_hp_hyper_next/results/score_logging_summary.csv`
- Score report: `experiments/v6_hp_hyper_next/reports/score_logging_report.md`

## 2. S1 Scorelog Anchor Result

S1 reran the anchor setting on:

- Subset: `hotpot_hard_1000`
- Seed: `42`
- Methods: `V4_topk2_fixed`, `V5_topk2_fixed`, `V6_HP_hyper_anchor`
- Payload target: `0.070134 ± 0.002`
- `topk=2`, `warmup=0`, `score_mode=value`, `budget_mode=fixed`, `layerwise_budget=True`

Downstream metrics reproduced the previous B2 pattern:

| Method | Payload | MRR | NDCG | F1 | EM | Recall@3 | Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4_topk2_fixed | 0.070134 | 0.8358 | 0.8113 | 0.7185 | 0.6400 | 0.8575 | 0.9240 |
| V5_topk2_fixed | 0.070134 | 0.8358 | 0.8113 | 0.7185 | 0.6400 | 0.8575 | 0.9240 |
| V6_HP_hyper_anchor | 0.070134 | 0.8372 | 0.8119 | 0.7200 | 0.6400 | 0.8575 | 0.9250 |

Conclusion:

- Score logging did not change payload or downstream metrics.
- The instrumentation is safe for continued diagnosis.

## 3. Score-distribution Findings

The new score logging produced 150 records:

- 3 methods.
- 10 rounds.
- 5 clients per round.
- 50 score records per method.

Summary:

| Method | Records | Block Jaccard vs V6 | JS Divergence vs V6 | Pooler Ratio | Encoder.layer.8 Ratio | Entropy | Avg Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hypernet_v4 | 50 | 1.0000 | 0.0624 | 0.5000 | 0.4600 | 0.6931 | -1.0179 |
| hypernet_v5 | 50 | 1.0000 | 0.0624 | 0.5000 | 0.4600 | 0.6931 | -1.0179 |
| hypernet_v6 | 50 | 1.0000 | 0.0000 | 0.5000 | 0.4600 | 0.6931 | -1.5925 |

Interpretation:

- V4/V5/V6 selected the same block set: `encoder.layer.1`, `encoder.layer.2`, `encoder.layer.8`, `encoder.layer.9`, and `pooler`.
- V4/V5 and V6 do have different internal score distributions, shown by JS divergence `0.0624` against V6.
- However, the selected block Jaccard remains `1.0`, so score-distribution differences are not large enough to change top-k selection.
- Pooler is selected in 50% of selected slots because `pooler` is effectively always uploaded.
- `encoder.layer.8` occupies 46% of selected slots, confirming strong layer-8 dominance.
- The negative selected-vs-next margin suggests that at least part of the selected set is not selected purely because of score rank. In practice, the always-upload `pooler` rule can force a block into the upload set even when another non-selected block has a higher score.

## 4. RQ Answers So Far

RQ1: Do V4/V5/V6 have different internal score distributions, even though selected blocks are identical?

Yes. V4/V5 differ from V6 in score distribution, with JS divergence around `0.0624`. But this difference does not change the selected block set.

RQ2: Is `topk=2` too restrictive?

Likely yes. The same selected block set appears across V4/V5/V6 despite non-identical score distributions. This suggests the selection bottleneck is dominated by a small number of high-priority or forced blocks.

RQ3: Is pooler / encoder.layer.8 dominance suppressing selection diversity?

Yes, current evidence supports this diagnosis. Pooler ratio is `0.50`, encoder.layer.8 ratio is `0.46`, and entropy remains low at `0.6931`.

RQ4: Does `layerwise_budget` actually constrain selection?

Not yet answered. S1 only reran the layerwise-on anchor. The next required ablation is `layerwise_budget=True` vs `False`.

RQ5: Can downstream_value, hard-query weighting, pooler cap, or same-payload adaptive reallocation produce meaningful selection diversity?

Not yet answered. These S2 ablations are prepared but not run.

## 5. Current Decision on B3 / Seeds / Adaptive

B3 `hotpot_hard_500`: still deferred.

Seeds 43/44: still deferred.

Adaptive large matrix: still deferred.

Reason:

- S1 confirms that V4/V5/V6 selection collapse remains real.
- V6 metric improvement is still too small for expansion.
- Score logging shows score-distribution differences, but not meaningful selection diversity.
- The next step should be targeted S2 ablation, not broader benchmark expansion.

## 6. Recommended Next Step

Priority order:

1. Run `GROUP=pooler` ablation first, because current scorelog directly implicates pooler dominance.
2. Run `GROUP=layerwise` to test whether layerwise budget actually changes selection.
3. Run `GROUP=score_mode` to test whether `downstream_value`, `delta`, or `grad_norm` can break selection collapse.
4. Only after one group creates selection diversity should we consider B3 or seed expansion.

Suggested command when GPU is available:

```bash
GROUP=pooler bash experiments/v6_hp_hyper_next/run_selection_diversity_ablation.sh
```

## 7. Current Paper-facing Claim

Confirmed:

Under strict same-payload constraints, V6-HP-hyper preserves HotpotQA hard-query performance at payload around `0.0701`, but current fixed `topk=2` selection causes V4/V5/V6 to converge to identical upload block sets.

New evidence:

V4/V5/V6 are not necessarily identical internally. Their score distributions can differ, but the combination of low `topk`, pooler always-upload, and encoder.layer.8 dominance prevents those score differences from becoming different upload decisions.

Not supported:

Current evidence still does not support a claim that V6-HP-hyper significantly outperforms V3/V4/V5 under same-payload constraints.

Promising but unverified:

Pooler dominance control and selection-diversity-aware low-budget selection are now the most promising next mechanisms to test.
