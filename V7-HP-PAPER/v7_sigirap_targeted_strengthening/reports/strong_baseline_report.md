# Strong Independent Reranker Baseline

## Protocol

CrossEncoder-Top5 independently scores each document in the same frozen approximately Top-10 candidate pool and retains five documents. It uses the same frozen cross-encoder checkpoint, 3,200-character cap, FLAN-T5-Large prompt and decoding, sentence-support predictor, and official metrics as Full. It excludes pair features, missing-hop features, outcome models, selector probabilities, gold labels, and reader outcomes at inference.

The ordering variant `ce_score_order` was chosen using the 1,000-query nested development split only. Both 3,000- and 3,405-query results are frozen evaluations of that choice. Because this baseline was added after the primary study, it is a **post-hoc secondary baseline analysis**, not a pre-specified confirmatory comparison.

## Results

### Development (nested 1,000)

Baseline / CrossEncoder / Full Joint F1: 0.3241 / 0.3300 / 0.3305. CrossEncoder minus Full is -0.0005 (95% CI [-0.0168, +0.0154], paired p=0.9668).

### Original holdout (3,000)

Baseline / CrossEncoder / Full Joint F1: 0.3292 / 0.3420 / 0.3356. CrossEncoder minus Full is +0.0064 (95% CI [-0.0033, +0.0156], paired p=0.1884).

### Revision holdout (3,405)

Baseline / CrossEncoder / Full Joint F1: 0.3201 / 0.3405 / 0.3280. CrossEncoder minus Full is +0.0124 (95% CI [+0.0034, +0.0211], paired p=0.0068).

## Cost boundary

Latency artifact status: `complete`. Frozen Top-5 and Full remain fixed at 140.88 and 213.48 ms/query. Direct same-machine CrossEncoder-Top5 mean/median/P95 latency is 149.90/135.47/262.59 ms/query; CrossEncoder scoring alone averages 11.33 ms/query. The direct scoring path reproduces 100.0% of cached Top-5 contexts.

## Claim boundary

This analysis asks whether independent document relevance can recover the pair-complementary result under one frozen pool and reader. It does not establish universal superiority over neural reranking or over other retrievers.
