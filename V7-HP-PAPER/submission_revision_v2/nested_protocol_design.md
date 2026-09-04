# Fully Nested Query-Level Protocol

## Purpose

The original v2.3 feature `safe_answer_prob` excluded the target query from its own estimator but pooled outcomes from other queries in the same outer test fold. That design was useful diagnostically, but it was not a strict held-out deployment estimate. The submission-v2 protocol rebuilds all outcome-derived features and all selector choices inside each outer training split.

## Outer split

The 1,000 unique query IDs are sorted by the integer value of their MD5 hash. Outer test fold `i` is `ordered_ids[i::5]`; each fold contains 200 test queries and 800 train queries. The exact IDs are stored in `fold_manifest.json`.

For each outer fold:

1. Construct outcome labels only for the 800 outer-train queries.
2. Split outer train into five deterministic query-level inner folds.
3. Fit the answer-safety nuisance model on four inner folds and predict the held-out inner fold. Concatenating these predictions produces outer-train OOF `safe_answer_prob` values.
4. Fit the nuisance model once on all outer-train queries and predict outer-test actions.
5. Train candidate selector configurations using only outer-train OOF features and outcomes.
6. Select thresholds, action-family restrictions, and coverage using only outer-train queries.
7. Freeze the nuisance model, selector, thresholds, action scope, and coverage.
8. Predict one action or fallback for each outer-test query. No test outcome is returned to training.

## Information boundary

The following are unavailable to decision-time feature extraction for an outer-test query:

- gold answer and target answer correctness;
- gold supporting titles or sentences;
- `doc.is_support` or oracle bridge labels;
- reader outcomes for any outer-test action;
- oracle action identity;
- held-out threshold or coverage feedback.

Allowed features are query-document lexical statistics, retrieval/policy scores, bridge proxies, action shape, anchor preservation, and the outer-train-fitted nuisance prediction.

## Primary objective

The primary method is a constrained two-stage selector rather than the inherited weighted utility:

1. predict answer safety and reject actions below the train-selected safety threshold;
2. among surviving actions, score the probability of an answer-safe evidence/product improvement;
3. apply only the top train-selected fraction of query actions;
4. otherwise preserve the frozen baseline context.

All five folds selected the same primary configuration from outer-train data: two-stage scoring, 0.5 coverage, conservative effective action families, safety threshold 0.5, and positive threshold 0.1. This agreement was observed after training; it was not imposed from test outcomes.

## Audits

`nested_feature_audit.json` records zero overlap for every outer train/test split and every inner train/validation split. It also records `outer_test_outcome_used=false` for all folds. `nested_fold_configs.json` contains the full train-only configuration traces.

## Primary outputs

- 1,000 per-query held-out decisions: `nested_per_example_delta.jsonl`;
- aggregate metrics: `nested_final_1000_summary.json`;
- 2,000-resample paired bootstrap: `nested_significance_report.json`;
- safety/support/weighted-objective ablations: `nested_ablation_summary.json`.

The protocol is fully nested with respect to the existing action table and outcome labels. It does not make those action labels deployable training data; it estimates the held-out behavior of a supervised organizer trained on outcomes from other queries.
