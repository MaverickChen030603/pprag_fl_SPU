# Seed and Checkpoint Manifest

## Global seeds

| Purpose | Seed |
| --- | ---: |
| Generator and selector training | 20260714 |
| Support predictor and primary bootstrap | 20260714 |
| SIGIR diagnostic bootstrap | 20260715 |

## Frozen pretrained checkpoints

| Role | Checkpoint identifier |
| --- | --- |
| MPNet semantic encoder | `sentence-transformers/all-mpnet-base-v2`, snapshot `e8c3b32edf5434bc2275fc9bab85f82640a19130` |
| CrossEncoder relevance | `cross-encoder/ms-marco-MiniLM-L-6-v2`, snapshot `c5ee24cb16019beea0893ab7796b1df96625c6b8` |
| Primary answer reader | `google/flan-t5-large`, snapshot `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` |
| Secondary answer reader | UnifiedQA-T5-Large frozen artifact (Answer-only robustness) |

## Fold selector checkpoints

| Fold | Safe threshold | Utility threshold | Coverage | Model SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.6 | 0.3 | 0.30 | `2ccb4a55cbb70343a3da98d5b6b6b872bc95d29973c0ff6c2a80b95274e29e04` |
| 1 | 0.5 | 0.3 | 0.30 | `d55c5667faf22d5279c8d7bedbfce818005b7c870c3a992db8ee25f8864b6f7f` |
| 2 | 0.5 | 0.3 | 0.25 | `6983a6fd8269ebcd56fc84344fb03a524e7117b398ab827a02e0c2bc0367b77c` |
| 3 | 0.6 | 0.3 | 0.15 | `533c59ab3e3d7055f5dcf53551855eacb34d1198354cd7d16ac648cccc4742d6` |
| 4 | 0.6 | 0.3 | 0.30 | `81ca2b24a819d0ebe04f31348f29a8462c047c204117fb95187cba491bb9d7cc` |

Frozen original-holdout selection artifact SHA-256: `8988faa66eca4451f3773656aabaa953c87de0fc9d0595c9c0d74f0c526ece51`.

Absolute cache and user paths are intentionally omitted from manuscript-facing artifacts.
