# V17 Data Locality Audit Specification

## Phase A

Phase A is a centralized offline Oracle audit of a simulated federated knowledge environment. Gold labels are used only in dispersion analysis and outcome evaluation. Therefore Phase A must not be described as federated model training.

## Phase C Audit, Conditional on Continuation

Before any FL result is accepted, the audit checks:

1. client dataloaders expose only origin-assigned queries;
2. server aggregation accepts tensors, counts, and approved aggregates only;
3. serialized updates contain no strings, query IDs, answers, contexts, or generated text;
4. personal adapters are excluded from aggregation payloads;
5. central audit copies are inaccessible from training model code;
6. communication bytes are measured from serialized payloads;
7. no privacy claim exceeds the implemented mechanism.

Current status: `not_applicable_before_checkpoint_A`.
