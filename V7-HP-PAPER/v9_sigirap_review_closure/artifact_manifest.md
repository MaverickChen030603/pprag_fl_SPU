# Artifact Manifest

Repository-relative paths are used to preserve anonymity. SHA-256 values describe the local review-closure source artifacts.

| Artifact | Role | SHA-256 |
| --- | --- | --- |
| `opportunity_aware_semantic_generation_v4/outputs/scaleup/frozen_selector_manifest.json` | Frozen result or audit evidence | `1b122a390ebdfd9fb8323b2ef68f069b2b78625d1a76a6df9e4da85845e10071` |
| `opportunity_aware_semantic_generation_v4/outputs/scaleup/official_metrics/scaleup_official_summary.json` | Frozen result or audit evidence | `6a58dbdb2740dcdd61f4fbf5ab93fec6d268f6c4b6a0afa65e0ebecf02e28c09` |
| `review_driven_revision_v5/outputs/lite_model/lite_holdout_metrics.json` | Frozen result or audit evidence | `1ad5524a4e1ab5ea7ae6a768c7b387d32ec8438075c4e2d8ef72f671faa31eb7` |
| `v7_sigirap_targeted_strengthening/outputs/reranker/ce_reranker_metrics.json` | Frozen result or audit evidence | `b05a5a28248d051f37d8673c9bdecd7080779a0945047318ae40a3df201e74b5` |
| `v7_sigirap_targeted_strengthening/outputs/reranker/ce_action_build_manifest.json` | Frozen result or audit evidence | `ee514995dd08e74c5a3f9fe02c93b72b073cf8148dcf1e61c70d946e226488da` |
| `v7_sigirap_targeted_strengthening/outputs/oracle/oracle_metrics.json` | Frozen result or audit evidence | `99434cca6a00759f6a88aa79dd02bc2b7aaacd09120b81c84ea8029fa1f74bf9` |
| `v4_submission_completion/outputs/generator_ablation/generator_ablation_results.json` | Frozen result or audit evidence | `1fb0d8e697cddee32156c1a9decf18d00c771d2ddd72f3bfcac7a12791bdeacb` |
| `v4_submission_completion/outputs/generator_ablation/generator_ablation_preparation_audit.json` | Frozen result or audit evidence | `285cdb6abc0e303ba1e4edc12d27583fe3e8b6643e3bc8f1b7f91cef7703a561` |
| `v5_final_review_optimized_submission/outputs/cost/frozen_latency_full_v4.json` | Frozen result or audit evidence | `8e0d5c4807842489d7a9f683600d55f160dc00cbf4a9288aaddfb5a3520c2b23` |
| `v5_final_review_optimized_submission/outputs/cost/frozen_latency_frozen_top5_baseline.json` | Frozen result or audit evidence | `f4b3e70aefb249c0f570627fd5d1145c8c6a578a666e2f7b77fa64c338abac08` |
| `v7_sigirap_targeted_strengthening/outputs/reranker/ce_reranker_latency.json` | Frozen result or audit evidence | `23705880eeb54f5a496f71b757f7aae8e6fbaa6c672b23ab4dc0dffcac7982d4` |

## Review-closure scripts

- `01_end_to_end_secondary_ablations.py`: `e02c18781181ca969daef102b3d5ebf55dc282b789a6da9097bc4f02af6ae013`
- `02_support_threshold_sensitivity.py`: `1ee6b1e9cd30a8e7fccb3648654c1ef185aa6fa4566a1f139c3718c8cd1a5b87`

The support sensitivity run emits its own manifest and leaves `primary_threshold=0.7`, `threshold_retuned=false`, `contexts_changed=false`, and `answer_outputs_changed=false`.
