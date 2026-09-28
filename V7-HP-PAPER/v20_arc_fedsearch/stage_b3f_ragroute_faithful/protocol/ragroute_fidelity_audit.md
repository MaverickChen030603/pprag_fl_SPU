# RAGRoute Paper--Code Fidelity Audit

**Stage:** B3F-0  
**Retrieved:** 2026-09-28 (JST)  
**Status:** `ragroute_architecture_resolved_from_official_history`  
**Permitted name after this audit:** `RAGRoute (protocol-adapted)`, not “exact reproduction”.

## Scope and source-of-truth

This audit precedes any B3F training or C1/C2 evaluation. It freezes the evidence used to adapt the official RAGRoute source-wise router to this project's fixed Top-8 candidate pool and multi-hop QA retrieval contract. It does not modify ProbeRoute, C1, or C2.

| Priority | Frozen source | Evidence |
|---|---|---|
| 1 | Official historical repository code | `sacs-epfl/ragroute`, historical training commit `a374c83` and historical router implementation |
| 2 | Published paper | *Efficient Federated Search for Retrieval-Augmented Generation using Lightweight Routing*, EuroMLSys 2025; arXiv `2502.19280` |
| 3 | Official current-main repository | HEAD `77c163f` |
| 4 | This project's future glue code | Only candidate-pool, label, corpus and downstream-stack adaptation |

The official repository was retrieved from `https://github.com/sacs-epfl/ragroute` at HEAD `77c163f14855e3b412891fc97339986f0a640d79`. The downloaded official paper is frozen in `official_source/ragroute_paper.pdf`; exact file hashes are recorded in `official_source/ragroute_source_manifest.json`.

## Architecture discrepancy resolution

The task briefing anticipated a conflict between a paper architecture `256 -> 128 -> 1` and a newer `128 -> 64 -> 32 -> 1` implementation. The official evidence resolves the issue rather than leaving it ambiguous:

| Source | Architecture | Finding |
|---|---|---|
| Published paper, §4.1.1 | `input -> 256 -> 128 -> 1` | Two hidden blocks: LayerNorm, ReLU, Dropout; one raw-logit output. |
| Historical official `a374c83`, `scripts/train/train_feb4rag_router.py` | `input -> 256 -> 128 -> 1` | Matches paper architecture, dropout 0.4. |
| Historical official `a374c83`, `ragroute/router.py` | `input -> 256 -> 128 -> 1` | Matches paper inference architecture. |
| Current official `77c163f`, training/router | `input -> 128 -> 64 -> 32 -> 1` | Post-paper implementation change. |

**Decision.** B3F will use `256 -> 128 -> 1`, LayerNorm/ReLU/Dropout(0.4) after each hidden layer. This is resolved from paper-era official training and inference code, not selected by current-task performance. No architecture sweep is allowed.

## Itemized fidelity matrix

| Item | Official evidence | B3F decision | Status |
|---|---|---|---|
| Router architecture | Paper §4.1.1; historical `a374c83` | `D -> 256 -> 128 -> 1` | Resolved |
| Hidden normalization/activation/dropout | Paper; historical train/router | LayerNorm, ReLU, Dropout 0.4 | Resolved |
| Output | Paper; code | One raw logit | Resolved |
| Input semantics | Current/historical `router.py` | query embedding + source centroid + global source-ID one-hot | Resolved |
| Query encoder | Official system uses source-compatible encoders | B3F uses the frozen BGE router embedding shared with existing B3 centroids; local retrieval remains unchanged | Protocol adaptation |
| Source centroids | Official `router.py` | Mean local-corpus embeddings in the router embedding space; freeze SHA-256 | Protocol adaptation |
| Source-ID encoding | Official `router.py` | 20-dimensional global client-ID one-hot, never Top-8 rank | Protocol adaptation |
| StandardScaler | Paper §4.1.1; official MedRAG/MMLU code | Fit on B3F training rows only, transform validation/evaluation only | Paper-priority completion |
| Training label | Official source-wise relevance objective | Same support-presence label as M2: client contains >=1 canonical support document | Controlled task adaptation |
| Loss | Paper; historical FeB4RAG training | `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` | Resolved |
| Optimizer | Historical FeB4RAG training | Adam, lr 0.001, weight decay `1e-5` | Resolved |
| LR schedule | Paper; historical FeB4RAG training | CyclicLR 0.001--0.005, triangular2, step_size_up=10, cycle_momentum=False; preserve official post-cycle StepLR transition | Resolved |
| Gradient clipping | Historical FeB4RAG training | max norm 1.0 | Resolved |
| Epochs | Historical FeB4RAG training | 150; cyclic before epoch 115 then StepLR(50, 0.05) | Resolved |
| Model selection | Paper; historical FeB4RAG script | Best validation accuracy (the script variable is misnamed `best_f1`) | Resolved |
| Validation split | Official paper uses query-level split | Deterministic query-level Router-Dev split within 5,000 historical Probe-Train IDs; C1/C2 excluded | Protocol adaptation |
| Threshold | Official current/historical inference | OriginalThreshold selects `p > 0.5`; no fallback unless official evidence exists | Resolved |
| Fixed3 inference | Required matched-budget adaptation | Sort independent probabilities; deterministic global client-ID tie break; select exactly 3 | Protocol adaptation |
| Reranking/retrieval stack | Official system differs | Excluded from matched-budget B3F; use frozen local depth 10, 5 docs/client, raw dense merge Top-10, Reader Top-5 | Required protocol adaptation |

## Important evidence limits

The paper says StandardScaler, while the historical FeB4RAG training script visible at `a374c83` does not explicitly apply it; official MedRAG and MMLU training paths do. B3F therefore follows the published method description and records this as a paper-priority implementation completion. The source does not expose a single universal, paper-era checkpoint/config bundle for this project's data, so B3F cannot be called an exact reproduction.

The official FeB4RAG script's stated “best F1” variable is updated by validation **accuracy**. B3F will reproduce the behavior, document the naming defect, and also report precision, recall, F1, AUC, true positive rate and predicted positive rate for sanity checking.

## Frozen B3F contract before implementation

1. Train only on the restored 5,000-query historical Probe-Train IDs and their frozen P0 Top-8 candidates. Do not use C1, C2, R5, development, answer, Reader, or reward labels.
2. Construct one row per query-client pair using BGE query embedding, frozen client centroid, global 20-client one-hot, and support-presence target.
3. Use three training seeds (0, 1, 2); report mean ± standard deviation. Never choose the best seed. Fixed3 and threshold predictions must use the predeclared three-seed probability mean.
4. Produce both Fixed3 and OriginalThreshold from the same averaged router probabilities. The former is the matched 3-client/15-document comparison; the latter is a dynamic-cost reference only.
5. Create an untouched B3F-Fresh split before any target labels are read. If no eligible pool exists, stop with `fresh_baseline_eval_unavailable` rather than silently relabeling C1/C2 as confirmatory.

## Fidelity gate decision

The architecture ambiguity gate is **passed**. B3F may proceed to feature/centroid reproduction and historical validation. Formal evaluation is blocked until the following are frozen: training/validation manifests, centroid hashes, model/scaler/config hashes, B3F-Fresh zero-overlap audit, and a label-free blind-evaluation plan.
