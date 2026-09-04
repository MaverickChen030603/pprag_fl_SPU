# Method Identity Audit Report

## Scope

- Raw CSV inputs: experiments/v6_hp_hyper_next/results/same_payload_baseline_raw.csv, experiments/v6_hp_hyper_next/results/same_payload_b2_hard1000_raw.csv
- Audited runs: 8
- Score distribution fields are not available in current round logs, so score means/stds are marked as not recorded.

## Selection Summary

| subset | seed | method | unique block sets | top blocks | layer distribution | utility memory | hard weighting |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| hotpot_all_1000 | 42 | V3_topk2_fixed | 4 | pooler:50;encoder.layer.8:45;encoder.layer.6:2;encoder.layer.7:2;encoder.layer.2:1 | `{"encoder.layer.2": 1, "encoder.layer.6": 2, "encoder.layer.7": 2, "encoder.layer.8": 45, "pooler": 50}` |  |  |
| hotpot_all_1000 | 42 | V4_topk2_fixed | 4 | pooler:50;encoder.layer.8:46;encoder.layer.9:2;encoder.layer.2:1;encoder.layer.1:1 | `{"encoder.layer.1": 1, "encoder.layer.2": 1, "encoder.layer.8": 46, "encoder.layer.9": 2, "pooler": 50}` | False | True |
| hotpot_all_1000 | 42 | V5_topk2_fixed | 4 | pooler:50;encoder.layer.8:46;encoder.layer.9:2;encoder.layer.2:1;encoder.layer.1:1 | `{"encoder.layer.1": 1, "encoder.layer.2": 1, "encoder.layer.8": 46, "encoder.layer.9": 2, "pooler": 50}` | False | True |
| hotpot_all_1000 | 42 | V6_HP_hyper_anchor | 4 | pooler:50;encoder.layer.8:46;encoder.layer.9:2;encoder.layer.2:1;encoder.layer.1:1 | `{"encoder.layer.1": 1, "encoder.layer.2": 1, "encoder.layer.8": 46, "encoder.layer.9": 2, "pooler": 50}` | False | True |
| hotpot_hard_1000 | 42 | V3_topk2_fixed | 4 | pooler:50;encoder.layer.8:45;encoder.layer.6:2;encoder.layer.7:2;encoder.layer.2:1 | `{"encoder.layer.2": 1, "encoder.layer.6": 2, "encoder.layer.7": 2, "encoder.layer.8": 45, "pooler": 50}` |  |  |
| hotpot_hard_1000 | 42 | V4_topk2_fixed | 4 | pooler:50;encoder.layer.8:46;encoder.layer.9:2;encoder.layer.2:1;encoder.layer.1:1 | `{"encoder.layer.1": 1, "encoder.layer.2": 1, "encoder.layer.8": 46, "encoder.layer.9": 2, "pooler": 50}` | False | True |
| hotpot_hard_1000 | 42 | V5_topk2_fixed | 4 | pooler:50;encoder.layer.8:46;encoder.layer.9:2;encoder.layer.2:1;encoder.layer.1:1 | `{"encoder.layer.1": 1, "encoder.layer.2": 1, "encoder.layer.8": 46, "encoder.layer.9": 2, "pooler": 50}` | False | True |
| hotpot_hard_1000 | 42 | V6_HP_hyper_anchor | 4 | pooler:50;encoder.layer.8:46;encoder.layer.9:2;encoder.layer.2:1;encoder.layer.1:1 | `{"encoder.layer.1": 1, "encoder.layer.2": 1, "encoder.layer.8": 46, "encoder.layer.9": 2, "pooler": 50}` | False | True |

## Pairwise Identity Checks

### hotpot_all_1000 / seed=42
- V3_topk2_fixed vs V4_topk2_fixed: identical_selection=False
- V3_topk2_fixed vs V5_topk2_fixed: identical_selection=False
- V3_topk2_fixed vs V6_HP_hyper_anchor: identical_selection=False
- V4_topk2_fixed vs V5_topk2_fixed: identical_selection=True
- V4_topk2_fixed vs V6_HP_hyper_anchor: identical_selection=True
- V5_topk2_fixed vs V6_HP_hyper_anchor: identical_selection=True

### hotpot_hard_1000 / seed=42
- V3_topk2_fixed vs V4_topk2_fixed: identical_selection=False
- V3_topk2_fixed vs V5_topk2_fixed: identical_selection=False
- V3_topk2_fixed vs V6_HP_hyper_anchor: identical_selection=False
- V4_topk2_fixed vs V5_topk2_fixed: identical_selection=True
- V4_topk2_fixed vs V6_HP_hyper_anchor: identical_selection=True
- V5_topk2_fixed vs V6_HP_hyper_anchor: identical_selection=True

## Current Audit Conclusion

V3/V4/V5 do not appear to have fully identical selection signatures in the audited runs.
Because block score distributions are not recorded, the current audit can verify selected block/layer identity but cannot compare score distribution identity.
