# V17 Experiment Status

**Current phase:** Phase-A client partition construction  
**Current scientific status:** no method claim before Federated Oracle Checkpoint A  
**Final test:** frozen at 2,000 queries per dataset; labels sealed read-only and unused  
**Model training:** prohibited before Checkpoint A

## Completed

- Primary-source Federated RAG and reader-aware selection search.
- Closest-collision and provisional naming audit.
- V17 directory and output contract.
- Draft frozen protocol and data-locality contract.
- V1-V16 exposure inventory, including V16's 9,000 exposed queries per dataset.
- New disjoint train/development/calibration/final splits for all three datasets.
- Frozen-split audit: zero prior-exposure overlap, zero cross-split overlap, valid hashes, and sealed final labels.
- Strict client-local retrieval implementation and six passing contract tests.

## In Progress

- Natural and random client partitions.
- Evidence dispersion and federated Oracle implementation.

## Guardrail

Positive V16 centralized Oracle StrictSyn is not evidence that federation helps. V17 must measure federation-induced opportunity on new, non-overlapping queries and stop if Checkpoint A fails.
