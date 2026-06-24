# V6-HP-hyper Paper Landing Diagnosis Report V2

Date: 2026-06-22 JST

## 1. Current Evidence

Confirmed:

- B1/B2 same-payload benchmarks are complete.
- S1 scorelog anchor rerun is complete.
- V6-HP-hyper preserves HotpotQA hard-query performance at payload approximately `0.0701`.
- V4/V5/V6 select the same block set under the current fixed `topk=2` low-budget configuration.
- Score logging shows that V4/V5 and V6 have different internal score distributions, but these differences do not change selected blocks.

Current scorelog evidence:

- Selected-block Jaccard vs V6: `1.0` for V4 and V5.
- JS divergence vs V6: approximately `0.0624` for V4/V5.
- Pooler selected ratio: `0.50`.
- Encoder.layer.8 selected ratio: `0.46`.
- Layer entropy: `0.6931`.

## 2. Why V6 Superiority Is Not Yet Supported

Not supported:

- Current evidence does not support a strong claim that V6-HP-hyper significantly outperforms V4/V5 under strict same-payload constraints.
- V6 improves B2 hard_1000 only slightly: MRR `+0.0014`, F1 `+0.0015`, Hit@10 `+0.0010`.
- EM and Recall@3 do not improve.

Interpretation:

The issue is not simply downstream difficulty. The hard-query subset is harder, but V4/V5/V6 still collapse into the same selected upload decisions.

## 3. Selection Collapse Evidence

Confirmed:

- V4/V5/V6 selected-block Jaccard is `1.0`.
- The shared selected block set is:
  - `encoder.layer.1`
  - `encoder.layer.2`
  - `encoder.layer.8`
  - `encoder.layer.9`
  - `pooler`

Likely cause:

- Low `topk=2`.
- Pooler always-upload.
- Encoder.layer.8 dominance.
- Layerwise budget currently does not yet prove enough diversity under the anchor setting.

## 4. Score-distribution Evidence

Confirmed:

- V4/V5 and V6 have non-identical internal score distributions.
- However, selected-block Jaccard remains `1.0`.

Interpretation:

The scoring mechanisms are not fully identical internally, but the current selector bottleneck prevents score differences from becoming different upload decisions.

## 5. Pooler Dominance Ablation

Current status:

- Pooler ablation has been prepared and queued on the server.
- It has not completed yet because all GPUs are currently occupied.
- Queue PID: `583103`.
- Queue log: `experiments/v6_hp_hyper_next/logs/pooler_ablation_queue_20260622_021439.log`.

Planned configurations:

- `v6_no_pooler_cap`
- `v6_pooler_cap_25`
- `v6_pooler_cap_10`
- `v6_pooler_exclude`

Expected outputs:

- `experiments/v6_hp_hyper_next/results/pooler_ablation_raw.csv`
- `experiments/v6_hp_hyper_next/results/pooler_ablation_summary.csv`
- `experiments/v6_hp_hyper_next/reports/pooler_ablation_report.md`
- `experiments/v6_hp_hyper_next/logs/pooler_ablation_commands.log`

Decision rule:

- If pooler cap creates selection diversity without metric drop, pooler-controlled selection should become the next paper-facing mechanism.
- If pooler exclude causes a large metric drop, keep it as diagnostic only.

## 6. Layerwise Budget Ablation

Current status:

- Not started.
- It should only run after pooler ablation completes.

Purpose:

- Test whether `layerwise_budget=True` actually changes selected layers.
- Test whether `layerwise_budget=False` increases pooler/layer8 dominance.

## 7. Score Mode Ablation

Current status:

- Not started.
- It should only run after pooler and layerwise groups complete.

Purpose:

- Test whether `downstream_value`, `delta`, or `grad_norm` can break value-score collapse.
- If score distributions change but selected blocks remain unchanged, the main bottleneck is likely forced pooler plus topk=2.

## 8. Hard-query Weighting Ablation

Current status:

- Not started.
- Lower priority than pooler/layerwise/score_mode because the strongest current evidence points to pooler/layer8 dominance.

## 9. Adaptive Reallocation Pilot

Current status:

- Not started.
- Should only run if pooler, layerwise, or score_mode ablation produces meaningful selection diversity.

## 10. Whether B3 / Seeds Are Justified

B3 hard_500: deferred.

Seeds 43/44: deferred.

Reason:

- No completed ablation has yet produced meaningful selection diversity.
- Current V6 improvement remains below the threshold for broader expansion.
- B3 or seed expansion before selection diversity would likely only repeat the same collapse pattern.

## 11. Recommended Paper Claim

Current recommended claim:

Under strict same-payload constraints, V6-HP-hyper preserves HotpotQA FedRAG retrieval performance at payload approximately `0.0701`. However, current fixed `topk=2` low-budget selection causes V4/V5/V6 to converge to identical upload decisions. Score logging shows that internal score distributions differ, but pooler always-upload and encoder.layer.8 dominance prevent those differences from becoming different selected blocks. Therefore, the next paper-facing mechanism should focus on pooler-controlled or diversity-aware selective upload under the same payload budget.

## 12. Next Steps

1. Wait for the queued pooler ablation to start and finish safely.
2. Analyze `pooler_ablation_summary.csv`.
3. If pooler cap creates diversity without metric drop, update the paper mechanism around pooler-controlled low-budget selection.
4. If pooler cap fails to change selection, run layerwise and score_mode ablations next.
5. Continue deferring B3/seeds until selection diversity is demonstrated.
