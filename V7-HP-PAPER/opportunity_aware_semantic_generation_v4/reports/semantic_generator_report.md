# Semantic Generator Training Report

The generator uses frozen MPNet bi-encoder embeddings and a frozen MS MARCO cross-encoder, then learns three outer-fold-specific components from outer-train outcomes only: a missing-hop estimator, a semantic document opportunity model, and a pair-complementarity model.

- Outer folds: 5 x 800 train / 200 frozen test queries.
- Mean best inner-CV document average precision: **0.2308**.
- Mean best inner-CV pair average precision: **0.3274**.
- Target-query gold support, answer, reader outcomes, and oracle actions: **not used**.

Action generation remains pending until `04_generate_outer_fold_actions.py` applies each frozen fold model to its disjoint outer-test queries.


## Frozen Outer-Test Generation

- Queries: **1000**.
- Effective actions: **7934**.
- Actions not present in the v3 table: **5655**.
- Per-query action budget: learned no-intervention gate with a hard maximum of **8**.
- No-leak audit: **pass**.

The generator ranks full per-query distractor documents semantically, estimates pair complementarity, and then materializes a bounded set of six reader-compatible action types. V3 action membership is recorded only after generation for marginal-efficiency accounting and is not used in action scoring.
