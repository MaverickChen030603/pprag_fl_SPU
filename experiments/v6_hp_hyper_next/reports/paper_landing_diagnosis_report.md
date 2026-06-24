# V6-HP-hyper Paper Landing Diagnosis Report

Date: 2026-06-19 JST

## 1. Current Evidence Summary

The current V6-HP-hyper stage has completed HotpotQA hard-query subset construction, smoke tests, B1/B2 same-payload benchmarks, and method identity auditing.

Confirmed:

- The hard-query subset construction is valid and creates a harder evaluation setting than the general HotpotQA subset.
- Under strict same-payload constraints, V6-HP-hyper preserves FedRAG retrieval performance at payload approximately `0.0701`.
- Current fixed `topk=2` low-budget selection leads V4/V5/V6 to highly similar or identical upload patterns.

Not supported:

- Current evidence does not support a claim that V6-HP-hyper significantly outperforms V3/V4/V5 under same-payload constraints.
- Current evidence does not yet prove that V6's hard-query-aware or downstream-value-aware mechanisms are actively changing selected upload blocks.

Promising but unverified:

- Hard-query-aware scoring, downstream-value scoring, pooler dominance control, and same-payload adaptive reallocation may improve method-level selection diversity, but require targeted ablation and score-distribution evidence.

## 2. Same-payload B1/B2 Findings

B1 on `hotpot_all_1000` and B2 on `hotpot_hard_1000` were completed with aligned payload.

For B2 `hotpot_hard_1000`, all methods used:

- Payload approximately `0.070134`.
- `topk=2`.
- `warmup=0`.
- `score_mode=value`.
- `budget_mode=fixed`.
- `layerwise_budget=True`.

| Method | MRR | NDCG | F1 | EM | Recall@3 | Hit@10 | Payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3_topk2_fixed | 0.8358 | 0.8113 | 0.7185 | 0.6400 | 0.8575 | 0.9240 | 0.070134 |
| V4_topk2_fixed | 0.8358 | 0.8113 | 0.7185 | 0.6400 | 0.8575 | 0.9240 | 0.070134 |
| V5_topk2_fixed | 0.8358 | 0.8113 | 0.7185 | 0.6400 | 0.8575 | 0.9240 | 0.070134 |
| V6_HP_hyper_anchor | 0.8372 | 0.8119 | 0.7200 | 0.6400 | 0.8575 | 0.9250 | 0.070134 |

V6 shows a weak positive delta:

- MRR: `+0.0014`.
- NDCG: `+0.0006`.
- F1: `+0.0015`.
- Hit@10: `+0.0010`.

This is not strong enough for a paper-level superiority claim.

## 3. Method Identity Audit

The current method identity audit shows:

- V3 has a different selected-block signature.
- V4/V5/V6 have identical selected-block signatures on both `hotpot_all_1000` and `hotpot_hard_1000`.

For `hotpot_hard_1000`:

- V3: `pooler:50; encoder.layer.8:45; encoder.layer.6:2; encoder.layer.7:2; encoder.layer.2:1`.
- V4/V5/V6: `pooler:50; encoder.layer.8:46; encoder.layer.9:2; encoder.layer.2:1; encoder.layer.1:1`.

This means that the downstream metric gap is constrained by upload-decision equivalence.

## 4. Selection Collapse Diagnosis

Current evidence suggests that the problem is not simply insufficient downstream difficulty. The hard-query subset is harder, but V4/V5/V6 still collapse to the same selected upload pattern.

Likely causes:

- `topk=2` is too restrictive and allows dominant blocks to suppress alternative scoring mechanisms.
- `pooler` and `encoder.layer.8` dominate selection across clients/rounds.
- `layerwise_budget=True` does not currently create enough selection diversity under the fixed low-budget setting.
- `score_mode=value` may be too close across V4/V5/V6 when utility memory is disabled.

## 5. Score-distribution Logging Results

Score-distribution logging has now been added to V4, V5, and V6-HP1 code paths. The new logging records per-round/per-client candidate score summaries, selected score summaries, selected-vs-non-selected margins, pooler selection counts, encoder.layer.8 selection counts, and layer entropy.

Current status:

- Logging code is implemented.
- Aggregation script is implemented.
- S1 scorelog anchor rerun is completed on `hotpot_hard_1000`, seed 42.
- Score logging produced 150 records: 3 methods, 10 rounds, 5 clients per round.
- V4/V5 differ from V6 in internal score distribution, with JS divergence around `0.0624`.
- Selected block Jaccard remains `1.0`, so internal score differences do not yet translate into selection diversity.
- Pooler selected ratio is `0.50`; encoder.layer.8 selected ratio is `0.46`, confirming strong pooler/layer-8 dominance.

Output targets:

- `experiments/v6_hp_hyper_next/results/score_logging_raw.jsonl`
- `experiments/v6_hp_hyper_next/results/score_logging_summary.csv`
- `experiments/v6_hp_hyper_next/reports/score_logging_report.md`

## 6. Selection-diversity Ablation Results

Selection-diversity ablation scripts have been prepared but not executed yet because all GPUs are currently occupied by other workloads.

Prepared ablation groups:

- Layerwise budget on/off.
- Score mode: `value`, `downstream_value`, `delta`, `grad_norm`.
- Hard-query weighting strength: off/default/strong/very strong.
- Pooler dominance control: no cap, cap 25%, cap 10%, exclude pooler.
- Same-payload adaptive reallocation pilot.

The key evaluation criterion is not only metric improvement, but whether selected upload patterns become meaningfully different from the V6 anchor while respecting payload constraints.

## 7. Which Mechanism Actually Changes Selection?

Confirmed:

- V3 changes selection relative to V4/V5/V6.

Not supported yet:

- It is not yet supported that V6-specific hard-query weighting or downstream-value scoring changes selection under the current anchor configuration.

Promising but unverified:

- Pooler control may directly test whether `pooler` dominance causes collapse.
- `delta` and `grad_norm` score modes may help determine whether the collapse is specific to value-density scoring.
- Layerwise on/off can test whether layerwise budget is an active constraint or a mostly inactive regularizer under `topk=2`.

## 8. Which Mechanism Improves Hard-query Metrics?

Confirmed:

- Current V6 anchor improves B2 hard-query MRR/F1 only slightly.

Not supported:

- No current ablation has proven a meaningful hard-query metric improvement.

Promising but unverified:

- If pooler cap or downstream-value scoring creates selection diversity without hurting payload, it may become the next paper-facing mechanism.

## 9. Whether to Run B3 / Multi-seed / Adaptive

B3 `hard_500` is deferred.

Reason:

- Current methods still lack meaningful selection diversity under same-payload constraints.
- V6 B2 metric improvement is below the threshold for expansion.
- Running B3 or seeds 43/44 now would likely only confirm the same collapse pattern.

Do not launch B3 or multi-seed unless at least one condition is met:

- An ablation produces selected-block Jaccard vs anchor `<= 0.85` and MRR/F1 improves by at least `0.005`.
- `downstream_value`, hard-query weighting, or pooler cap creates a clearly different selection signature even if metrics are temporarily flat.
- Same-payload adaptive reallocation creates different selection while keeping payload within `0.070134 ± 0.002`.

## 10. Recommended Paper Claim

Recommended current claim:

Under strict same-payload constraints, V6-HP-hyper preserves HotpotQA FedRAG retrieval performance at a very low payload around `0.0701`. However, the current fixed `topk=2` configuration causes V4/V5/V6 to converge to nearly identical upload selections, limiting the observable downstream advantage of V6. This motivates a selection-diversity diagnosis and mechanism-level ablation rather than blind benchmark expansion.

Claims to avoid:

- Do not claim that V6 significantly outperforms V3/V4/V5 under same-payload constraints.
- Do not claim that adaptive or hard-query-aware V6 is effective until score logging and ablations show that the mechanism changes selection.

## 11. Next Experiment Plan

Immediate next steps:

1. Wait for a safe GPU window or use a queued idle-GPU launcher.
2. Run `run_scorelog_anchor_hard1000.sh` for V4/V5/V6 anchor rerun.
3. Collect `score_logging_raw.jsonl`, `score_logging_summary.csv`, and `score_logging_report.md`.
4. Run targeted ablation groups one at a time:
   - `GROUP=layerwise`
   - `GROUP=score_mode`
   - `GROUP=hard_weight`
   - `GROUP=pooler`
   - `GROUP=adaptive`
5. Run `run_method_identity_audit_after_ablation.py` after each group.
6. Decide whether B3/multi-seed is justified based on selection diversity and hard-query metric delta.
