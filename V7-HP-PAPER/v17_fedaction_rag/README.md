# V17 Federated Context Action Selection

V17 tests whether knowledge fragmentation across clients creates enough cross-client-only context repair opportunity to justify a federated action selector. The internal engineering label is `FedAction-RAG`; the formal name is not frozen.

The project distinguishes:

1. a federated knowledge environment, where documents and local retrieval are client-scoped;
2. federated model training, where raw training queries and reader outcomes remain local.

Phase A is an Oracle opportunity audit only. No centralized selector, FedAvg model, FedProx/SCAFFOLD model, or personalized adapter may be trained until the preregistered Federated Oracle checkpoint passes.

## Phase-A Order

1. merge the V1-V16 exposure inventory;
2. freeze new V17 train/development/calibration/final splits;
3. build topic, entity-community, Dirichlet, and random partitions;
4. assign training-query origins without gold evidence;
5. audit gold-evidence dispersion offline;
6. create strict client-local retrieval pools;
7. run centralized and federated Oracle action evaluation;
8. issue `federated_go_no_go_phase_a.md` before any selector training.

Final labels remain sealed throughout development. Data locality is an algorithmic contract, not a formal privacy guarantee.
