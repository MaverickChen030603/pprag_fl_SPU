# V17 Preregistration: Federated Context Action Opportunity

**Frozen before Phase-A reader outcomes:** 2026-07-22  
**Experiment:** V7-HP-PAPER-v17-federated-action-rag  
**Internal label:** FedAction-RAG  
**Primary stage:** Federated Oracle feasibility, not learned-method evaluation

## 1. Primary Question

Do natural knowledge silos increase cross-client composition-only reader gains relative to a centralized pool and size-matched random document partitions, under fixed client and context budgets?

## 2. Data and Exposure

Datasets are HotpotQA, 2WikiMultiHopQA, and MuSiQue. V17 excludes every ID and normalized question found in the V1-V16 inventory, including all V16 train, development, calibration, and final-test inputs. New split sizes per dataset are:

| Split | Size | Use |
|---|---:|---|
| train | 5,000 | future local action labels, only if Phase A passes |
| development | 1,000 | Phase-A Oracle and architecture development |
| calibration | 1,000 | future gates, only after earlier checkpoints |
| final_test | 2,000 | sealed confirmatory evaluation |

The final labels are separated from final inputs and not read before the method and statistical contract are frozen.

## 3. Frozen Retrieval and Reader Contract

- document encoder: `BAAI/bge-base-en-v1.5`;
- sparse retriever: SQLite FTS5 BM25;
- hybrid score: 0.55 dense + 0.45 min-max sparse;
- answer readers: `google/flan-t5-large` and `allenai/unifiedqa-v2-t5-large-1363200`;
- support/evidence evaluator: frozen V16 support predictors;
- one final answer-reader call per candidate context in offline Oracle labeling;
- no gold/local-support injection and no random padding.

## 4. Client Partitions

Primary client count is `M=20`. Topic silos are the single primary natural partition for the machine Go/No-Go decision; entity-community silos are a preregistered natural-partition replication. Random balanced partition is the negative control. Dirichlet partitions with alpha 0.1, 0.3, and 1.0 are stress tests, not natural-silo evidence. This priority was frozen before any V17 reader outcome was generated.

Partition construction cannot use queries, answers, supporting facts, or reader outcomes. Each document belongs to exactly one client. Main analyses also report client-size imbalance and partition entropy.

## 5. Query Origin

Query origin is assigned from query-to-client topic similarity with deterministic stochastic sampling and seed 20260723. Gold evidence is not available to assignment. The origin client remains included in routed client sets.

## 6. Frozen Budgets

- primary client budget: `B_c=3`;
- secondary feasibility budget: `B_c=2`;
- local retrieval depth: `k=5`;
- final context budget: `K=5`;
- primary client count: `M=20`;
- query-all is an upper bound, not the main operating point.

Main federated pools contain the origin client's top five plus the highest-scoring documents returned by the other routed clients, truncated to ten documents for exact matched Oracle search. The centralized control uses the same query, ten-document pool size, five-document output budget, readers, and search code.

## 7. Phase-A Sample Sizes

An exploratory smoke uses at most 20 queries per dataset-reader-partition cell and cannot trigger a Go decision. The machine checkpoint requires at least 100 queries in every primary cell:

- three datasets;
- two readers;
- centralized, topic, entity-community, and random conditions;
- `M=20`, `B_c=3`, `k=5`, `K=5`.

The `B_c=2` analysis may use the same 100 queries but is a secondary feasibility requirement.

## 8. Oracle Definitions

For metric M:

`FedCompGain_M = BestCrossComposition_M - max(BestSingleClient_M, BestSingleCrossAction_M)`

A cross-client composition-only query satisfies:

1. `BestCrossComposition_M > 0`;
2. `BestSingleClient_M <= 0`;
3. `BestSingleCrossAction_M <= 0`.

`CrossClientStrictSyn`, `WithinClientStrictSyn`, `RandomPartitionStrictSyn`, and the same-query centralized V16-style StrictSyn are reported separately. MuSiQue uses a constructed Answer-Evidence composite and is never called official Joint.

## 9. Primary Statistical Tests

The primary test compares cross-client composition-only rates and mean `CrossClientStrictSyn` for each natural partition against both centralized and random controls on matched queries.

- paired bootstrap: 5,000 resamples;
- confidence interval: 95%;
- two-sided paired tests;
- effect size reported with each comparison;
- BH correction across secondary partition/budget comparisons;
- minimum 100 queries per primary dataset-reader-condition cell.

## 10. Checkpoint A

FedAction-RAG training is allowed only if all conditions hold:

1. at least two datasets and both readers have mean CrossClientStrictSyn >0 with CI lower bound >0;
2. at least two datasets have cross-client composition-only rate >=10% on both readers;
3. the natural-partition composition-only rate is significantly higher than same-query centralized and random-balanced controls;
4. gains concentrate in queries whose gold evidence is actually dispersed across clients;
5. a realizable opportunity remains with `B_c<=3`, with positive secondary evidence at `B_c=2` or a documented client-recall explanation;
6. the two readers agree in direction.

If the checkpoint fails, no centralized selector, FL selector, or personalized adapter is trained. The project becomes `federated_opportunity_analysis_only` or `oracle_gap_not_supported`.

## 11. Later Checkpoints

Phase B begins only after A passes and tests centralized learnability. Phase C begins only after B passes and compares Local, FedAvg, FedProx, SCAFFOLD, shared-only, shared-plus-adapter, and centralized pooled models. All Phase-B/C hyperparameters must be frozen in an amendment written before their outcomes.

## 12. Prohibited Post-hoc Actions

- changing partition type after inspecting reader outcomes;
- increasing `B_c`, pool size, or local k to rescue the primary checkpoint;
- selecting only MuSiQue if HotpotQA/2Wiki fail;
- using gold evidence in routing or candidate generation;
- calling raw-data locality privacy preservation;
- training a selector before the machine-readable Phase-A decision.
