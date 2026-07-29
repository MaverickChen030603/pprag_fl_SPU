# Checkpoint-A Freeze and V18 Entry Contract

## Frozen V17 execution

The formal run is identified by Git commit `fac9f62` and the following fixed
contract:

| Item | Frozen value |
|---|---|
| Queries per dataset-reader-condition cell | 100 |
| Datasets | HotpotQA, 2WikiMultiHopQA, MuSiQue |
| Readers | FLAN-T5-Large; UnifiedQA-T5-Large |
| Conditions | centralized; topic-silo Bc=2; topic-silo Bc=3; entity-community Bc=3; random-balanced Bc=3 |
| Natural primary partition | topic-silo |
| Client count / budget | M=20 / Bc=3 primary, Bc=2 secondary |
| Local retrieval / final context | k=5 / K=5 |
| Local sparse candidate cap | 20 |
| Reader calls | identical offline Oracle trajectory contract |
| Decision metrics | CrossClientStrictSyn and composition-only as frozen in V17 |

Until all thirty cells are complete, V18 may only add static documentation,
tests, source code that is not executed against V17 outcomes, literature
audits, and recovery/checksum support. It must not inspect intermediate reader
outcomes to modify any of the values above.

## Permitted post-completion branches

### Strong pass

At least two datasets pass on both readers, Bc=3 has positive StrictSyn with a
CI lower bound above zero, Bc-realizable cross-client-only rate is at least
10%, natural silos exceed both matched controls, Bc=2 agrees in direction, and
the effect concentrates in genuinely dispersed evidence. V18 can enter
centralized learnability.

### Conditional pass

The evidence is restricted to one dataset, a reader, high-dispersion cases,
or Bc=3. V18 may only conduct a preregistered Phase-A2: a larger development
sample and an inference-safe high-dispersion opportunity detector. The method
claim becomes a targeted slow-path composer, not a universal federated model.

### Fail

If natural silos do not exceed random controls, cross-client-only opportunity
is scarce, reader directions conflict, or evidence dispersion does not explain
the signal, stop FedAction composition. Preserve the source-routing and
communication audit, and redirect to reader-aligned selective retriever-block
upload without retrofitting partitions or budgets.

## Later checkpoint requirements

Phase B needs a centralized opportunity-gated composer to beat Fast Path and
the best learned single edit, realize at least 30% of the oracle composition
gap, and remain non-dominated in quality/cost. Phase C then compares
local-only, FedAvg, FedProx, SCAFFOLD, shared-only, and shared-plus-local
adapters over alpha 0.1/0.3/1.0 with at least three seeds. Phase D evaluates
payload-matched selective model-block upload only if Phase C is effective.
