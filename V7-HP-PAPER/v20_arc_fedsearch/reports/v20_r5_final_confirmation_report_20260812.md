# V20 R5 Sealed Final Confirmation Report

## Executive Decision

**Final status:** `final_test_strongly_confirmed`  
**Lifecycle status:** `v20_empirical_evaluation_complete`

V20 method development is closed. The frozen logistic ProbeRoute improves Joint F1 over the inherited federated baseline on all three held-out datasets and both frozen readers. The six paired 95% confidence intervals are strictly positive. The macro Joint F1 gain is **+0.0332**, compared with **+0.0387** in R4, and all six R4-to-R5 directions agree.

The R5 sample is a V17 train-derived, previously untouched held-out split (`N=300` per dataset), not an official hidden test. Selection was completed without labels by a preregistered hash rule.

## Frozen Protocol

- Datasets: HotpotQA, 2WikiMultiHopQA, MuSiQue; `N=300` each.
- Methods: M0 federated baseline, M1 label-free ProbeRoute, M2 logistic ProbeRoute, M3 centralized retrieval reference.
- ProbeRoute: candidate Top-8, 18 float32 probe features/client, 592 B/query, select 3 clients.
- Retrieval: local depth 10, 5 documents/client, 15 transmitted documents, raw global Top-10, reader Top-5.
- Readers: frozen FLAN-T5-Large and UnifiedQA-T5-Large.
- Statistics: 5,000 query-paired bootstrap resamples; Joint F1 primary, Answer/SP secondary with BH-FDR.
- SP uses one shared frozen context-level support predictor and is not an independent reader replication.

## Retrieval Context Completion

Reader-context complete support at Top-5 is reader-independent.

| Dataset | M0 Baseline | M1 Label-free | M2 Logistic | M3 Centralized |
|---|---:|---:|---:|---:|
| HotpotQA | 0.4300 | 0.5633 | **0.5767** | 0.5800 |
| 2WikiMultiHopQA | 0.1700 | 0.2300 | **0.2467** | 0.2767 |
| MuSiQue | 0.1167 | **0.1733** | **0.1733** | 0.2100 |

M2 raises complete reader-visible evidence by +14.67 pp on HotpotQA, +7.67 pp on 2Wiki, and +5.67 pp on MuSiQue relative to M0. On HotpotQA, M2 nearly reaches the centralized reference under the fixed federated communication contract.

## Dual-Reader Main Results

### FLAN-T5-Large

| Dataset | Method | Answer F1 | SP F1 | Joint F1 |
|---|---|---:|---:|---:|
| HotpotQA | M0 | 0.5689 | 0.3493 | 0.2536 |
|  | M1 | 0.6117 | 0.3949 | 0.2876 |
|  | **M2** | **0.6226** | **0.4002** | **0.2906** |
|  | M3 | 0.6121 | 0.3866 | 0.2803 |
| 2Wiki | M0 | 0.3803 | 0.2776 | 0.1379 |
|  | M1 | 0.3934 | 0.3721 | 0.1758 |
|  | **M2** | 0.3906 | **0.3873** | **0.1799** |
|  | M3 | 0.4294 | 0.3988 | 0.2150 |
| MuSiQue | M0 | 0.2165 | 0.3053 | 0.0784 |
|  | M1 | **0.2482** | 0.3598 | 0.0971 |
|  | **M2** | 0.2476 | **0.3669** | **0.1075** |
|  | M3 | 0.2803 | 0.3781 | 0.1241 |

### UnifiedQA-T5-Large

| Dataset | Method | Answer F1 | SP F1 | Joint F1 |
|---|---|---:|---:|---:|
| HotpotQA | M0 | 0.5240 | 0.3493 | 0.2335 |
|  | **M1** | **0.5622** | 0.3949 | **0.2621** |
|  | M2 | 0.5610 | **0.4002** | 0.2615 |
|  | M3 | 0.5713 | 0.3866 | 0.2599 |
| 2Wiki | M0 | 0.3142 | 0.2776 | 0.1219 |
|  | M1 | 0.3174 | 0.3721 | 0.1498 |
|  | **M2** | **0.3358** | **0.3873** | **0.1624** |
|  | M3 | 0.3299 | 0.3988 | 0.1656 |
| MuSiQue | M0 | 0.1763 | 0.3053 | 0.0678 |
|  | M1 | **0.2013** | 0.3598 | 0.0849 |
|  | **M2** | 0.2004 | **0.3669** | **0.0903** |
|  | M3 | 0.2380 | 0.3781 | 0.0990 |

M3 is a reference operating point rather than a formal upper bound. M2 exceeds M3 on HotpotQA Joint F1 for both readers, while M3 remains stronger on 2Wiki and MuSiQue.

## Primary Confirmatory Tests: M2 vs M0 Joint F1

| Dataset | Reader | Delta | 95% CI | p-value | Win/Tie/Loss |
|---|---|---:|---:|---:|---:|
| HotpotQA | FLAN | +0.0370 | [0.0172, 0.0582] | 0.00040 | 32/257/11 |
| HotpotQA | UnifiedQA | +0.0280 | [0.0121, 0.0442] | 0.00080 | 28/266/6 |
| 2Wiki | FLAN | +0.0420 | [0.0214, 0.0636] | 0.00040 | 42/245/13 |
| 2Wiki | UnifiedQA | +0.0405 | [0.0191, 0.0632] | 0.00040 | 39/246/15 |
| MuSiQue | FLAN | +0.0291 | [0.0124, 0.0469] | 0.00120 | 27/265/8 |
| MuSiQue | UnifiedQA | +0.0225 | [0.0059, 0.0405] | 0.00440 | 23/267/10 |

The primary confirmation is therefore positive across datasets and readers, rather than being driven by one benchmark or one reader.

## Secondary Outcomes

- SP F1 improves significantly in all six cells after BH-FDR (`q=0.00111`).
- HotpotQA Answer F1 improves for FLAN (+0.0537, `q=0.00111`) and UnifiedQA (+0.0371, `q=0.00864`).
- Answer F1 changes on 2Wiki are positive but not significant after FDR.
- MuSiQue Answer F1 is positive for both readers, but does not pass BH-FDR (`q=0.0533` FLAN; `q=0.1146` UnifiedQA).
- There is no systematic answer harm.

## Mechanism and Generalization

- Complete-support transitions under M2: **86 rescues vs 2 harms**.
- Mean Joint F1 gain within support-rescue cases: **+0.2695**.
- R4-to-R5 direction agreement: **6/6 cells**.
- R4 macro Joint delta: +0.0387; R5 macro Joint delta: +0.0332.
- Positive dataset count: 3/3.

These results support the mechanism claim that compact query-conditioned resource evidence improves client selection, which increases complete multi-hop evidence exposure and transfers to reader-backed Joint F1.

## Execution Audit

The run was interrupted during unlabeled centralized retrieval by server maintenance. Before shutdown, immutable packets and protocol files were checkpointed locally and remotely. Recovery reused the 900 complete probe packets and rebuilt only non-resumable partial centralized outputs.

All 3,600 unlabeled contexts and both 3,600-row unscored reader files were completed and checksum-validated before the evaluator opened labels at `2026-08-12T12:07:15.147088+00:00`.

The first evaluator invocation wrote every scientific CSV, then failed while serializing a NumPy `int64` into the final JSON. A persistence-only finalizer generated the decision/report/checksum from the already-frozen CSVs. It did not reopen labels or recompute metrics. This recovery is recorded in `r5_persistence_recovery_audit.json`.

## Frozen Claim Boundary

Supported: under the frozen three-client and 15-document contract, ProbeRoute yields reproducible cross-dataset resource-selection, complete-evidence, and reader-backed Joint F1 gains.

Not supported: universal no-harm guarantees, zero-cost claims, formal privacy/security guarantees, centralized retrieval as an upper bound, or universal superiority of logistic over label-free routing.

No further V20 method tuning, threshold selection, or result-triggered reruns are permitted.
