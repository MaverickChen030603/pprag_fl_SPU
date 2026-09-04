# Same-Payload Baseline Report (B1 all_1000 seed=42)

- Scope: `hotpot_all_1000`, seed `42`, target payload `0.070134 ± 0.002`.
- Status: completed for V3/V4/V5/V6-HP-hyper anchor.
- Limitation: this is a single-split/single-seed B1 result, not the final multi-seed benchmark.

| Method | Payload | MRR | NDCG | F1 | EM | Recall@3 | Hit@10 | Delta MRR vs V6 | Delta F1 vs V6 | Payload mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V3_topk2_fixed | 0.070134 | 0.8923 | 0.8745 | 0.7985 | 0.7320 | 0.9100 | 0.9550 | -0.0007 | -0.0005 | False |
| V4_topk2_fixed | 0.070134 | 0.8923 | 0.8745 | 0.7985 | 0.7320 | 0.9100 | 0.9550 | -0.0007 | -0.0005 | False |
| V5_topk2_fixed | 0.070134 | 0.8923 | 0.8745 | 0.7985 | 0.7320 | 0.9100 | 0.9550 | -0.0007 | -0.0005 | False |
| V6_HP_hyper_anchor | 0.070134 | 0.8930 | 0.8745 | 0.7990 | 0.7310 | 0.9090 | 0.9540 | +0.0000 | +0.0000 | False |

## Interpretation

- All four methods reached the same payload region (`0.0701343`) and are within tolerance.
- On `all_1000`, V6-HP-hyper anchor is only marginally higher in MRR/F1 than V3/V4/V5, while EM/Recall@3/Hit@10 are slightly lower.
- This supports the concern that the standard all_1000 split is still not discriminative enough for strong method claims.
- The next meaningful comparison should prioritize `hard_1000` and `hard_500` before expanding seeds.

## Current Claim

Under strict same-payload constraints on `all_1000`, V6-HP-hyper preserves performance but does not yet show a practically large advantage over V3/V4/V5.
