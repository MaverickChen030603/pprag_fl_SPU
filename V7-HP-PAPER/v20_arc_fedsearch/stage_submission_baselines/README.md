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
