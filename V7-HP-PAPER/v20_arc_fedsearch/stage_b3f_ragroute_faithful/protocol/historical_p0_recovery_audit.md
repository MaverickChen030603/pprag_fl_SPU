# Historical P0 Candidate-Universe Recovery Audit

**Date:** 2026-09-28  
**Status:** `historical_p0_profile_recovered_and_validated`  
**Decision:** The historical P0 candidate universe may be used for B3F formal training after its profile assets and training manifest are frozen.

## Recovery question

The historical M2 manifest certifies 5,000 training query IDs and 40,000 candidate rows per dataset, but the corresponding query-level P0 candidate packets were pruned (`candidate_packets_recovered: false`). This audit first tested, and rejected, the hypothesis that the retrospective B3 full-client BGE centroids were the historical P0 profile. It then located the versioned frozen P0 profiles used by the pre-registered route stages and validated their exact ranking behavior against independent frozen C1/C2 packets.

## Frozen inputs

- Encoder: `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.
- Query encoding: official project frozen BGE CLS embedding with L2 normalization.
- Candidate centroids: `runs/ragroute_b3_r5_posthoc_20260916/centroids/<dataset>/source_centroids.npy`, each with its existing centroid manifest and SHA-256.
- Validation observations: the first five label-free C1 and first five label-free C2 Probe packets for each dataset (10 queries per dataset). Only question, P0 candidates and static scores were read. No answers, support labels, Reader outputs or performance metrics entered the calculation.
- Test: compare reconstructed full-20 centroid-dot-query scores and stable Top-8 ranking against the P0 candidate list and stored static scores already present in the frozen packets.

## Rejected reconstruction hypothesis

| Dataset | Queries checked | Exact Top-8 matches | Maximum absolute static-score error |
|---|---:|---:|---:|
| HotpotQA | 10 | 10 / 10 | 0.00191 |
| 2WikiMultiHopQA | 10 | 5 / 10 | 0.00738 |
| MuSiQue | 10 | 9 / 10 | 0.00283 |

This hypothesis is falsified: exact membership fails on 2Wiki and MuSiQue, and the score differences are too large to treat the B3 centroids as a hash-equivalent historical P0 profile. The observed agreement on some queries is not sufficient evidence of identity. These retrospective B3 centroids are therefore not used to reconstruct the P0 candidate universe.

## Recovered frozen P0 profile

The project contains the frozen P0 profiles used by the pre-registered routing stages. For each dataset, the asset exposes `profiles[client].p0_single_centroid`; the query encoder is `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, using its normalized CLS representation, and candidates are selected by a stable descending dot-product ranking. The source files and SHA-256 values are:

| Dataset | Profile SHA-256 |
|---|---|
| HotpotQA | `e3b4cafd26bbbdef77993cfd90d5065c88ef40c259a27409c3eb69a5e5937b36` |
| 2WikiMultiHopQA | `e2c601c8fe90878bc318d9f7e5761009cbf7f14b8675e70837e64306dafa41cb` |
| MuSiQue | `1ed9982cef993429d1c0c83984312d8df4be285f6c7eec1a627966c726d71930` |

## Full label-free validation

The recovered profile was tested on every label-free query in the independently frozen C1 and C2 packet sets: 1,000 queries per dataset (2 x 500). The validation reads only question text, stored P0 client lists, and stored static scores. It does not read answers, supporting-document labels, Reader predictions, or metrics. The immutable machine-readable result is `historical_p0_profile_validation.json`.

| Dataset | Queries checked | Exact Top-8 matches | Maximum absolute static-score error |
|---|---:|---:|---:|
| HotpotQA | 1,000 | 1,000 / 1,000 | 1.34e-6 |
| 2WikiMultiHopQA | 1,000 | 1,000 / 1,000 | 1.13e-6 |
| MuSiQue | 1,000 | 1,000 / 1,000 | 1.19e-6 |

The small score differences are float32-level serialization/compute variation; all candidate memberships and orderings are exact. This validates the ranking implementation and recovered P0 profile against a preserved, independent artifact. It does not use C1/C2 labels and does not make C1/C2 a B3F evaluation set.

## Integrity consequence

It remains scientifically invalid to use the old B3 centroids to regenerate a new historical Top-8 universe and describe it as the M2 universe. It would also be invalid to solve a new P0 centroid from C1/C2 query observations. Neither action is taken here: the profile existed as a frozen, versioned project asset before this B3F work and is validated only through label-free packet fields.

The recovery condition is now met by the second admissible path: a versioned P0 source-profile artifact, fixed BGE revision and pooling/ranking implementation, and exact validation against preserved independent packets. B3F training remains prohibited until the profile assets, recovered V17 IDs, query-level Router-Dev split, feature-centroid assets, and model configuration are committed in the training manifest. No B3F router checkpoint, route, Reader prediction, or result table exists at the time of this audit update.
