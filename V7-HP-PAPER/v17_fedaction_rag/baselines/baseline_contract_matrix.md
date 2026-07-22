# V17 Baseline Contract Matrix

This matrix freezes comparison contracts before reader outcomes are inspected. A published federated method is included numerically only when its official implementation can satisfy the same frozen-pool, client-budget, context-budget, and reader-call contract.

| Baseline | Client definition | Knowledge access | Query access | Retriever training | Selector training | Raw-data locality | Context budget | Reader calls | Communication | Privacy mechanism |
|---|---|---|---|---|---|---|---:|---:|---|---|
| Local-only RAG | Origin silo | Origin client only | Origin query | Frozen | None | Query and corpus local | 5 | 1 | One client response | None |
| Best single client Oracle | One of 20 silos | One oracle-selected client | Offline audit only | Frozen | None | Oracle diagnostic, non-deployable | 5 | Offline labels | One client response | None |
| Centroid routed union | Topic/entity/random silo | Origin plus top centroid clients | Label-free query embedding | Frozen local retrievers | None | Corpus local; query sent only to contacted clients | 5 | 1 | At most Bc local top-k responses | None |
| Query-all clients | 20 silos | Every client | Query broadcast | Frozen local retrievers | None | Corpora local | 5 | 1 | 20 local top-k responses | None |
| Flat union BM25 | Same contacted silos | Budget-matched union | Contacted clients | Frozen | Score sort only | Corpora local | 5 | 1 | Budget-matched documents | None |
| Flat union CrossEncoder | Same contacted silos | Budget-matched union | Aggregation server sees returned docs/query | Frozen local retrievers | Frozen relevance ranker | Raw corpora local; returned candidates centralized | 5 | 1 | Budget-matched documents plus scores | None |
| V14 Full adapted | Same contacted silos | Frozen union pool | Aggregation server sees returned candidates | Frozen | Offline reader-supervised selector | Training contract centralized | 5 | 1 | Budget-matched documents | None |
| Best single cross-client edit Oracle | Same contacted silos | Frozen union pool | Offline audit only | Frozen | None | Oracle diagnostic, non-deployable | 5 | Offline labels | Budget-matched documents | None |
| Cross-client composition Oracle | Same contacted silos | Frozen union pool | Offline audit only | Frozen | None | Oracle diagnostic, non-deployable | 5 | Offline labels | Budget-matched documents | None |
| Centralized hierarchical selector | Simulated silos | Frozen union pool | All training queries pooled | Frozen | Centralized reader-supervised | No training-data locality | 5 | 1 | Online retrieval only | None |
| FedAvg/FedProx/SCAFFOLD selector | Simulated silos | Frozen routed union | Training queries remain at origin | Frozen | Federated shared utility model | Raw queries/outcomes stay local | 5 | 1 | Retrieval plus model updates | None |
| Personalized FedAction-RAG | Simulated silos | Frozen routed union | Training queries remain at origin | Frozen | Shared federated model plus local adapter | Raw queries/outcomes and adapter stay local | 5 | 1 | Retrieval plus shared updates | None |

`Best single client`, single-edit, and composition Oracles measure opportunity and are not deployable baselines. No row claims differential privacy, secure aggregation, homomorphic encryption, or formal privacy.
