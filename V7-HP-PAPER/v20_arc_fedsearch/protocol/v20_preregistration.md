# V20 Preregistration: ARC-FedSearch

## Frozen Question

Under data-local and communication-constrained federated search, can adaptive
source routing, document-budget allocation, score calibration, and conditional
multi-hop source expansion recover more centralized-reference evidence utility
than fixed `Bc=3, local-k=5` retrieval at no higher mean communication cost?
The centralized retrieval run is a reference, not an oracle support upper bound.

## Stage U0

Before fitting a router, replay frozen V17 topic-silo development pools and
decompose losses into routing, local retrieval, transmission, calibration,
global truncation, context, and reader utilization.  Reader evaluation is
forbidden in U0.  Gold fields are available only after retrieval for offline
audit columns and oracle rows.

## Initial Baseline Contract

| Component | Baseline |
|---|---|
| Partition | V17 topic-silo, 20 clients |
| Router | origin client plus centroid-ranked sources |
| Contacts | fixed Bc=3 |
| Local retrieval | BGE + BM25 hybrid |
| Local depth | 5 documents per selected client |
| Merge | inherited raw hybrid score |
| Transmission | 15 documents |
| Global pool / context | Top-10 / Top-5 |

## Gates

- **A, loss audit:** identify a material and feasible routing, local retrieval,
  or merging loss.  If oracle client plus oracle merge still cannot approach
  centralized coverage, audit the partition contract instead of adding models.
- **B, retrieval smoke:** require at least +5 percentage points complete
  support at equal mean `Bc <= 3` and documents `<= 15`, more support gains than
  losses, and improved worst-hop rank.
- **C, multi-dataset:** require consistent complete-support improvement on at
  least two of HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- **D, reader:** run only after C, with FLAN-T5-Large and UnifiedQA-T5-Large.
