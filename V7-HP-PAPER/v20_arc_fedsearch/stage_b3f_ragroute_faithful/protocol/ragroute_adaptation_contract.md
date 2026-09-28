# B3F RAGRoute Protocol-Adaptation Contract

**Status:** `p0_universe_recovered_pending_training_manifest`  
**Issued:** 2026-09-28  
**Scope:** This contract governs B3F only. It does not alter M2, C1, C2, R5, or any revealed result.

## Intended comparison

The primary direct baseline will be **RAGRoute-Fixed3**: an official-paper-era source-wise RAGRoute router adapted only to this project's frozen candidate and downstream interfaces. It will score the same P0 Top-8 candidates as M2, select exactly three clients by descending probability (global client-ID ascending tie break), retrieve locally to depth 10, return five documents per selected client, merge raw dense Top-10, and construct the frozen Reader Top-5 context.

The companion **RAGRoute-OriginalThreshold** will use the same frozen router and select exactly the candidates satisfying `p > 0.5`. It will not force a three-client budget and will have no invented fallback. It is a cost/behavior reference, not a matched-budget significance baseline.

## Permitted information

For a query/client pair `(q, c)`, the router feature is

`[ BGE(q) ; centroid(c) ; one_hot_global_client_id(c) ]`.

No ProbeRoute 18-dimensional local signal, BM25 feature, local dense score, P0 rank, P0 static score, support statistic, Reader output, or C1/C2 label may enter the feature vector. Centroids are training-independent means of local-corpus BGE embeddings and are frozen by hash before training.

The supervised target is the controlled task adaptation

`y(q,c) = 1` iff client `c` contains at least one canonical supporting document for `q`.

It is the same source-presence supervision available to M2, but is not a Reader reward and does not claim word-for-word reproduction of the original paper's relevance labels.

## Training contract

- Query population: recovered historical V17/Probe-Train IDs, 5,000 queries per dataset.
- Candidate rows: exactly `5,000 × 8`; all P0 Top-8 non-support candidates retained as negatives.
- Architecture: `D -> 256 -> 128 -> 1`, LayerNorm/ReLU/Dropout(0.4), paper-era official implementation.
- Standardization: StandardScaler fit on training rows only.
- Loss: BCEWithLogitsLoss with `pos_weight = N_neg / N_pos`.
- Optimizer/schedule: Adam (`lr=1e-3`, `weight_decay=1e-5`); CyclicLR 0.001--0.005, triangular2, step_size_up=10, then official StepLR behavior after epoch 115; gradient norm cap 1.0; 150 epochs.
- Model selection: validation accuracy at deterministic query-level Router-Dev split; report other classification metrics but do not select on them.
- Seeds: 0, 1, 2; use their probability mean for both evaluation variants; report seed stability without selecting the best seed.

## Historical P0 recovery and remaining gate

The recovered historical manifest proves 5,000 training query IDs and 40,000 M2 candidate rows per dataset, while recording `candidate_packets_recovered: false`. The original query packet files remain unavailable. However, a versioned frozen P0 profile asset, exact BGE revision/pooling/ranking implementation, and independent label-free validation now recover the candidate universe. `historical_p0_recovery_audit.md` and `historical_p0_profile_validation.json` record this provenance; exact Top-8 order matched for all 1,000 C1/C2 packet queries in each dataset.

B3F **must not** use the previously tested and rejected retrospective B3 full-client centroids to define P0. It must use only the recovered frozen P0 profiles with the recorded hashes. Before training, the historical ID manifests, profile assets, routing feature centroids, query-level Router-Dev split, and all model settings must be committed in a machine-readable training manifest. Until that manifest exists, no model, route, Reader output, or B3F-Fresh result may be generated.

## Future blind evaluation requirement

After the training candidate universe is recovered and models/scalers/configurations are frozen, B3F-Fresh must be drawn from an untouched public pool with zero ID and normalized-question-hash overlap against Probe-Train, historical development, R5, C1 and C2. Only then may the label-free route/materialize/dual-Reader/hash sequence begin. If no such pool exists, the resulting C1/C2 comparison must be labeled post-hoc, not confirmatory.
