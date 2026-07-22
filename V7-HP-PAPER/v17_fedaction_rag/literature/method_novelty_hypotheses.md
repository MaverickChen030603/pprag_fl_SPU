# V17 Method Novelty Hypotheses

## Frozen Scientific Hypothesis

Federated knowledge fragmentation may increase the density of multi-hop repair opportunities that require complementary evidence from multiple clients. This is a testable hypothesis, not an assumed property of federated RAG.

## Candidate Contributions, Conditional on Evidence

### H-N1: Federation-induced opportunity

Natural topic/entity partitions produce more cross-client composition-only positive queries than both the original centralized Top-10 setting and client-size-matched random partitions.

### H-N2: Complete-context action contract

A selector chooses a complete ordered five-document action after budgeted client retrieval, rather than independently ranking clients or passages.

### H-N3: Federated opportunity decomposition

Failures are separated into routing absence, local retrieval absence, pool absence, single-action absence, composition-search absence, scoring miss, personalization miss, gating miss, and harmful realization.

### H-N4: Federated action-utility learning

Only after Oracle and centralized learnability pass, the shared action utility representation is federated while client-specific adapters remain local. Raw training queries, answers, contexts, and reader outputs are not sent to the aggregation server.

### H-N5: Quality-risk-communication selection

The final per-query decision jointly constrains client visits, final context cardinality, communication, and predicted reader harm, with exact baseline fallback.

## Claims Unavailable Regardless of Results

- first federated RAG system;
- first personalized Federated RAG;
- first source/client router;
- first reader-aware or set-wise context selector;
- privacy-preserving without implemented and audited privacy guarantees;
- end-to-end federated reader/retriever training;
- general multi-hop improvement from a single dataset or partition.

## Go/No-Go Interpretation

- If Phase A fails, the method novelty hypotheses are not supported and no selector is trained.
- If Phase A passes but centralized learning fails, the project becomes an Oracle/opportunity analysis.
- If centralized learning passes but FL does not, the conclusion is a federated optimization limitation, not a successful method.
- A method-paper claim requires natural-partition replication, random controls, two readers, budget matching, and final-test integrity.

## Provisional Formal Name

`Federated Context Action Selection for Multi-Hop RAG` is a descriptive placeholder. `FedAction-RAG` remains an internal engineering code until Phase C and a renewed name/collision audit.
