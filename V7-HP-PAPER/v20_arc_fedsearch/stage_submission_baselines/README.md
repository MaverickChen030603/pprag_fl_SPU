# ProbeRoute Submission Baselines

This stage fills the fixed-budget comparison gap without changing the frozen
V20 retrieval or Reader contract. The first run is intentionally marked
`retrospective_post_hoc_r5_revealed`: it may diagnose the already-revealed R5
holdout, but it cannot provide a new confirmation claim.

The frozen methods in this batch are:

- `b0_static_top3`: select the first three clients by the coordinator-side
  `static_candidate_rank`.
- `b1_random_top3_seed_00` through `b1_random_top3_seed_19`: select three
  clients from the same static Top-8 pool using
  `SHA256(seed|query_id|client_id)`.
- `b2_static_only_logistic`: train the same standardized, class-balanced
  Logistic protocol as M2 using only the coordinator-side static score. The
  equivalence audit determines whether its one-dimensional monotonic ranking
  makes it identical to B0 before any duplicate Reader run is scheduled.

The next, independently auditable external baseline is
`b3_ragroute_protocol_adapted`.  It follows the RAGRoute feature family
(query BGE embedding, full-client BGE source centroid, and client-ID one-hot)
with a shallow MLP.  It is trained only on a public non-R5 prefix with
support-client labels, then ranks clients only inside the already frozen R5
static Top-8.  It does not consume the 18-float probe.  `run_ragroute_b3.sh`
builds full (not sampled) source centroids, trains B3, materializes its
unlabeled contexts, invokes the two frozen Readers, and scores only after both
Reader output manifests validate.  The run remains a retrospective diagnostic
because R5 was already revealed.

Both baselines preserve local depth 10, five transmitted documents per selected
client, raw-score merge Top-10, and Reader Top-5. They do not transmit the
18-dimensional probe, so their routing probe cost is zero bytes.

Run the selection/materialization smoke first:

```bash
python materialize_posthoc_baselines.py \
  --input-root /path/to/inputs/r5_run \
  --index-root /path/to/inputs/indexes \
  --output-dir /path/to/run/materialized
```

The Reader and scoring stages are separate so that all unlabeled contexts are
frozen and hashed before any final-test label file is opened.

The Reader runner deduplicates identical `(dataset, query_id, context_hash)`
inputs and then maps the deterministic prediction back to every method/seed.
This changes neither prompts nor outputs and avoids repeated GPU work when
different random seeds produce the same Top-5 context.

## Completed R5 diagnostic run

The completed 2026-09-16 post-hoc run is summarized under
`results/posthoc_r5_20260916/`. The directory contains compact aggregate
tables, manifests, audits, and human-readable records only. Full materialized
contexts, per-query scores, Reader predictions, logs, and fitted model files
remain in the external experiment archive recorded in
`experiment_record_20260916.md`.
