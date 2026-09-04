# Federated Claim Decision

## Route chosen: Route B

The submission removes “Federated” from the title and treats Federated RAG as the motivating application, not as the experimentally validated system claim.

## Audit answers

1. **Does the evaluated candidate pool genuinely execute a federated protocol?** No. Candidate documents and previously computed routing scores are loaded from frozen artifacts.
2. **Is client identity natural?** No. After sanitization, documents are assigned `client_{index mod 5}`. This is a synthetic round-robin partition, not measured non-IID client ownership.
3. **Are texts centralized at the organizer?** Yes. The organizer can inspect every candidate text used to form an action.
4. **Do routing scores originate from prior FL experiments?** Some policy/retrieval metadata is inherited from the HP4 line, but the submission-v2 organizer does not rerun or optimize the federated communication pipeline.
5. **Is payload causally connected to the organizer intervention?** No direct end-to-end payload-to-organizer experiment is reported. Upstream payload diagnostics and downstream 50% action coverage are different quantities.
6. **Is real non-IID partitioning evaluated?** No.
7. **Are privacy, secure aggregation, or communication guarantees evaluated?** No.

## Consequence

Recommended title:

> **Reader-Safe Context Action Selection for Multi-Hop Question Answering**

Acceptable alternative:

> **Reader-Safe Evidence Organization under Distributed Candidate Retrieval for Multi-Hop QA**

The first is preferred because it maps exactly to the evaluated component. The introduction may explain that distributed or federated retrieval motivated the frozen candidate interface. It must also state that the organizer is centralized over exposed candidates, uses synthetic client IDs, and provides no privacy or communication guarantee.

## Permitted system diagram boundary

`distributed/federated candidate sources (motivation; frozen) -> centralized candidate pool -> supervised bounded action selector -> fixed reader`

Only the middle and downstream action-selection stages are evaluated by the main nested result.
