# Federated Data Flow Contract

## Knowledge Environment

Each client owns a disjoint document partition and a frozen local retriever. At inference, a router sends a query representation/request to at most `B_c` clients. Each contacted client returns at most `k` document records or controlled summaries. The server constructs one K-document context and invokes the frozen reader once.

## Federated Training, Conditional on Checkpoints A and B

Each origin client may read locally:

- its assigned training queries;
- permitted client-retrieval responses;
- offline reader outcome labels;
- local shared-model copy and personal adapter.

The aggregation server may receive:

- shared-model updates;
- aggregation weights;
- predefined aggregate optimization diagnostics.

The aggregation server may not receive as algorithm inputs:

- raw queries;
- gold answers or supporting facts;
- complete local training contexts;
- reader-generated answers;
- per-query outcome logs;
- local personal-adapter parameters.

The simulation may retain a central audit copy for reproducibility, but training entry points must reject it as a feature source.

## Claim Boundary

This contract supports the terms `data-local training`, `federated optimization`, and `raw-query/corpus locality`. It does not establish differential privacy, secure aggregation, homomorphic encryption, or formal privacy guarantees.
