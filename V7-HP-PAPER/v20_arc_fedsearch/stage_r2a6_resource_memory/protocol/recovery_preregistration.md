# V20 Stage R2-A.6: Representative Evidence Memory Profile Recovery Gate

## Scope

This stage evaluates candidate generation only. The deployable query-side input is the unmodified query encoded by frozen BGE; the resource-side input is a fixed, client-local Representative Evidence Memory Profile (REMP). Every profile is built from the documents that occur in `Router-Train`, with all answer, support, evidence, and query text fields ignored during profile construction.

The stage does not train a router or selector; it does not change `Bc=3`, local retrieval, document/query encoders, similarity function, or transmission contract. Reader, answer, joint metrics, dynamic client budgets, query views, LLM teachers, score calibration, and final test are prohibited.

## Frozen configuration

| Item | Value |
| --- | --- |
| Datasets | 2WikiMultiHopQA, MuSiQue |
| Partition / local corpus | inherited V17 `topic_silo`, 20 clients |
| Query / unit encoder | `BAAI/bge-base-en-v1.5`, normalized cosine |
| Client contact budget | `Bc=3` (evaluation metric only; no rerouting) |
| Candidate cutoffs | `K in {5, 8}` plus diagnostic `K=3` |
| REMP capacity | 32 units per client |
| Unit limits | title <=16 tokens; entity group <=12; snippet <=32; relation phrase <=12 |
| Deduplication | exact normalized text; token Jaccard >=0.85 near-duplicate exclusion |
| Pooling candidates | `S0=max`, `S1=mean(top-3)`, `S2=logsumexp` |
| Randomness | deterministic algorithms; two identical Recovery-Dev runs |

## Variants

- `B0`: inherited P0 single centroid.
- `B1`: inherited best K-means multi-prototype (`P8` for 2Wiki; `P16` for MuSiQue), with max prototype similarity.
- `R0`: 32 real document medoids of train-corpus clusters.
- `R1`: 32 farthest-first diverse train-corpus representatives.
- `R2`: 32 cross-client discriminative title/entity/snippet/relation units.
- `R3`: 16 R0 medoid units plus 16 R2 discriminative units.

No additional variants, learned weights, query views, or lexical rank fusion are allowed.

## Fresh splits and selection rule

`Recovery-Dev` is indices `[100, 200)` of the already frozen `Router-Dev`; `[0,100)` was used by R2-A/R2-A.5 and is excluded. `Recovery-Holdout` is the first 300 rows of the disjoint frozen `Router-Holdout`; it is not read before the Recovery-Dev decision.

On Recovery-Dev, all `R0..R3 x S0..S2` rows are compared against B0. Select exactly one shared `(strategy, pooling)` by this deterministic key among eligible choices:

1. largest minimum complete-set recall@5 delta across the two datasets;
2. then largest mean complete-set recall@5 delta;
3. then lower mean bytes per client;
4. then lexicographic strategy/pooling name.

A choice is eligible only if both datasets meet `complete-set@5 >= B0 + 0.05`, neither dataset lowers gold-client recall@5, neither lowers complete-set@8 by more than 0.01, and no reported primary metric degrades by more than 0.02. If no choice is eligible, Recovery-Holdout remains unread.

## Recovery-Holdout decision

If and only if Dev is eligible, freeze the selected strategy, pooling, profile capacity, thresholds, and seed. Evaluate the same configuration on each fresh holdout. Success requires both datasets to improve complete-set@5 by at least 5pp vs B0; at least one paired-bootstrap 95% CI lower bound must exceed zero and the other dataset must have a non-negative direction; CandidateAbsenceLoss@8 must decline; no client or small query subset may account for the gain.

The reader decision is always `blocked_before_reader` in this stage.
