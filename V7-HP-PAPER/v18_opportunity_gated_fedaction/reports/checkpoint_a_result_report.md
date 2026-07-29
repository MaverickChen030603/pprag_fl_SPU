# V17 Checkpoint-A Completion Report

**Protocol:** V7 Federated Action RAG Phase-A, `phase_a_checkpoint100`  
**Frozen execution commit:** `fac9f62`  
**Decision:** `checkpoint_a_fail` / `hold_or_redirect`  
**Permitted next step:** preserve the federated opportunity analysis and return
to the reader-aligned selective retriever-block upload line. Do not start
centralized composition learning, FedAvg/FedProx/SCAFFOLD, personalization, or
selective model-upload Phase D for FedAction.

## Integrity and scope

All 30 dataset-reader-condition cells completed at N=100. The V18 audit passed
104/104 checks: query-set alignment across conditions and readers, required
budgets, local-k, K=5 contexts, action-pool cap, document uniqueness, client-ID
validity, formal-start provenance, artifact hashes, and the static no-leak
audit. No final labels were accessed. There was no crash-recovery branch in
this first execution.

The contract compared centralized, topic-silo Bc=2, topic-silo Bc=3,
entity-community Bc=3, and random-balanced Bc=3 under the same 100 queries,
ten-document action pool, K=5 context budget, and two frozen readers.

## Primary Topic-Silo Bc=3 result

| Dataset | FLAN StrictSyn (95% CI) | FLAN comp-only | UnifiedQA StrictSyn (95% CI) | UnifiedQA comp-only | Result |
|---|---:|---:|---:|---:|---|
| HotpotQA | +0.0338 [-0.0098, +0.0753] | 7% | +0.0292 [-0.0078, +0.0648] | 4% | CI and rate fail |
| 2WikiMultiHopQA | -0.0342 [-0.0895, +0.0197] | 5% | -0.0078 [-0.0582, +0.0425] | 9% | direction/rate fail |
| MuSiQue | +0.0234 [-0.0248, +0.0703] | 10% | +0.0288 [-0.0133, +0.0716] | 11% | CI and centralized-control fail |

No primary dataset-reader cell has a StrictSyn CI lower bound above zero. No
dataset passes both readers under the preregistered conjunction, so the
machine decision correctly reports zero passing datasets.

## Controls and opportunity interpretation

Topic silos did create budgeted cross-client exposure: the cross-client
evidence rate was 43%/63%/74% for HotpotQA/2Wiki/MuSiQue, while complete
support in the Bc=3 action pool was 41%/20%/12%. This is a routing/opportunity
signal, not reader-synergy evidence.

Relative to random-balanced controls, topic-silo composition-only rate was
positive in several cells, notably MuSiQue FLAN (+10%, CI [+5%, +16%]) and
2Wiki UnifiedQA (+8%, CI [+3%, +14%]). However, topic-silo did not show the
required improvement over centralized controls: the composition-only
differences were 0/1% for HotpotQA, 0/1% for 2Wiki, and -1/0% for MuSiQue
(FLAN/UnifiedQA), with intervals crossing zero. Bc=2 was also not directionally
reliable. Entity-community partitions were mostly negative on StrictSyn.

The experiment therefore rejects the specific claim that the current natural
silos create sufficient *Bc-realizable, reader-beneficial* cross-client
composition opportunity for a general learned FedAction composer. It does not
reject source routing, communication auditing, or a future inference-safe
high-dispersion analysis, but these cannot be used to reopen the frozen Phase-A
claim by changing partitions or budgets.

## Artifacts

- V17 machine decision: `V7-HP-PAPER/v17_fedaction_rag/oracle/phase_a_checkpoint100/results/federated_go_no_go_phase_a.json`
- V17 aggregate and control tables: `federated_oracle_results.csv`,
  `partition_control_results.csv`, and `routing_metrics_summary.csv` in the
  same directory.
- V18 integrity: `checkpoint_a/checkpoint_a_integrity.json` and `.md`.
- V18 all-unit report: `checkpoint_a/checkpoint_a_all_units.csv`,
  `checkpoint_a_reader_consistency.csv`, and
  `checkpoint_a_partition_comparison.csv`.

## Recommendation

**`return_to_selective_upload`**. Retain this experiment as a negative but
informative federated opportunity analysis. Its main engineering value is the
audited, source-budgeted, two-reader evaluation contract; its scientific result
is that access dispersion alone did not survive the reader-level StrictSyn and
centralized-control tests. A targeted high-dispersion Phase-A2 should be
considered only under a newly frozen development protocol and only if its
inference-safe detector can be defined without gold evidence.
