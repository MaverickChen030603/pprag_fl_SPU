# V20 Stage M0-Confirm Frozen Contract

This document fixes the cross-dataset retrieval-only replay before execution.

| Field | Frozen value |
|---|---|
| Datasets | HotpotQA, 2WikiMultiHopQA, MuSiQue development only |
| Queries | Deterministic development rows 101--400 (`N=300`) |
| Route | Inherited V17 origin-plus-topic-centroid router |
| Client budget | `Bc=3` |
| Local candidate depth | 10 documents per physical client |
| Transmission budget | 15 documents |
| Global context pool | 10 documents |
| Primary allocation | A0 equal 5/5/5 |
| Primary merger | M1 rank percentile |
| Retriever/router | Frozen |
| Reader | Forbidden |

The all-client local-depth pass only materializes possible local candidates. It
does not alter `selected_clients`; every reported non-oracle metric is computed
over the inherited frozen Bc=3 route. Query text, frozen query origins, and
frozen centroid similarities are the only route inputs. Support labels and
answers are read only after construction for offline retrieval accounting.

Each dataset is run twice. The allocation matrix and per-query artifacts must
be byte-identical before their result is eligible for the cross-dataset gate.
