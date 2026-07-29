# V18 Opportunity-Gated FedAction

**Project:** `V7-HP-PAPER-v18-opportunity-gated-fedaction`  
**Internal name:** FedAction-RAG v2  
**Status:** preregistered preparation; V17 Checkpoint-A is running.

V18 is deliberately downstream of the frozen V17 Phase-A experiment. It does
not alter V17 retrieval, readers, query IDs, partitions, budgets, definitions,
or its final-test seal. This directory contains no learned composer or
federated training implementation until the V17 machine decision has been
integrity-audited.

## Decision order

1. Run `checkpoint_a/01_checkpoint_a_integrity_audit.py` only after all 30 V17
   reader-backed cells finish.
2. Produce the unified Phase-A table and issue `strong`, `conditional`, or
   `fail` status from the frozen decision contract.
3. Start Phase-A2 only for a preregistered conditional branch; start
   centralized learnability only after a strong pass or confirmed conditional
   pass. Neither path opens final-test labels.
4. Train personalized federated models only after centralized learnability;
   selective parameter upload is Phase D, never a Phase-A rescue mechanism.

The decision-object taxonomy and comparison contract are in
`literature/decision_object_matrix.md`. The V17 implementation remains the
source of truth for the running Checkpoint-A protocol.
