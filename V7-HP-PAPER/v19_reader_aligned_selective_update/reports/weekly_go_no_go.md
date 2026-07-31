# V19 Weekly Go/No-Go

## Week 1 initialization (2026-07-31)

- **Current gate:** Stage 0, full-upload viability.
- **Frozen inherited assets:** V17 split identities, final-test seal, topic-silo
  client assignments, routed development pools, FLAN-T5-Large, UnifiedQA-T5-
  Large, support evaluator, context budget, and V17 no-leak audit.
- **New implementation:** rank-8 LoRA retriever blocks (13 stable blocks),
  full-upload Frozen/Centralized/FedAvg/FedProx/SCAFFOLD smoke runner, and
  reader-compatible Top-5 context export.
- **Not implemented by design:** reader-credit surrogate, LCB, conflict-aware
  selection, knapsack, and private adapters. They are forbidden until Stage 0
  passes.
- **Decision:** `stage0_not_started`; no scientific claim is available yet.

## Next executable check

1. Generate schema and inherited V19 manifests.
2. Run HotpotQA development smoke with Frozen/Centralized/FedAvg/FedProx/
   SCAFFOLD using the same routed candidate pool.
3. Label each exported context using the frozen FLAN reader. If the adapter
   smoke is stable, repeat with UnifiedQA and then MuSiQue before the Stage 0
   gate is considered.
