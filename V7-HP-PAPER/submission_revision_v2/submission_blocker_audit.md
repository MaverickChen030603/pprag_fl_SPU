# Submission Blocker Audit

## Decision summary

The v1 manuscript was not submission-ready. Its answer-safety nuisance feature was query-excluded but not outer-fold nested, its `sp_f1` and `joint_f1` names could be mistaken for official HotpotQA metrics, the title overstated the evaluated federated setting, and the data construction was not documented as a reproducible manifest. The submission-v2 package resolves the leakage blocker with a five-fold fully nested rerun, reconstructs and verifies the 1,000-query sample, and narrows the paper to title-level evidence organization. Official sentence-level support evaluation and an end-to-end federated claim remain unsupported.

| Issue | Grade | v1 status | Submission risk | Resolution and v2 status |
|---|---|---|---|---|
| `safe_answer_prob` was not fully nested | P0 | Query-excluded over the full pool | Same-outer-fold outcomes could affect a held-out feature | **Resolved.** For every outer fold, the nuisance model is fitted only on outer-train queries; outer-train features are inner-OOF and outer-test features use a frozen outer-train fit. All train/test overlap audits are zero. |
| Non-official HotpotQA support/joint metrics | P0/P1 | Artifact names `sp_f1` and `joint_f1` were ambiguous | Official benchmark claim would be invalid | **Claim-level resolution.** Renamed to title-level support F1 and answer-title-support product. Official evaluation remains unavailable because predicted support sentence IDs were not produced. No official HotpotQA improvement claim is retained. |
| Data source and sampling manifest missing | P0 | Split and exact sampling could not be checked from the paper | Main sample was not reproducible | **Resolved.** Source is Hugging Face `hotpot_qa/distractor`, validation split; seed-44 shuffle and first 1,000 rows reproduce the stored ID sequence exactly. IDs, checksums, source reconstruction, and folds are exported. |
| Synthetic clients / centralized candidates | P1 | Federated title implied a stronger system than evaluated | Scope and privacy overclaim | **Resolved by Route B.** “Federated” is removed from the recommended title. Distributed/federated retrieval is motivation; the evaluated organizer is centralized and client IDs are synthetic. |
| Single reader | P2 | FLAN-T5-large only | Reader dependence | Retained as an explicit limitation; no reader-robust claim. |
| Failed 2Wiki transfer | P2 | Near-zero transferred selector gain | Cross-dataset generalization unsupported | Kept only as an appendix failure boundary; no transfer claim. |
| Proxy external baselines | P1/P2 | Local SetR-/RankRAG-style heuristics | Could be read as exact reproductions | Moved to appendix and labeled conceptual proxies. No “outperforms SetR/RankRAG” statement. |
| Utility weights were weakly justified | P1 | Weighted sum mixed product, answer, and support terms | Post-hoc and double-counting risk | **Resolved for the primary method.** Primary v2 uses a two-stage answer-safety gate followed by positive-action scoring, with fallback. The inherited weighted objective is diagnostic only; train-only sensitivity is exported. |
| Selected fraction = 0.5 | P1 | Could appear tuned on the final 1,000 outcomes | Hidden coverage tuning | **Resolved for the primary estimate.** Coverage/configuration are selected on each outer-train split and frozen on outer test. The 0.1-1.0 curve is diagnostic and is not used to replace the nested primary setting. |
| Citation placeholders / incorrect metadata | P0 | Placeholder citations and incorrect recent-work metadata | Manuscript cannot be submitted | **Resolved in v2 bibliography/report.** Only verified primary publication or arXiv records are retained; proxy comparisons are not converted into literature claims. |
| Environment / commands incomplete | P1 | Reader settings partly known; no complete package | Weak artifact review | **Partly resolved.** Commands, package versions, GPU, commit, prompts, seeds, and manifests are provided. Exact reader/tokenizer Hub revisions and historical runtime were not logged and remain `[NEEDS SOURCE FILE]`. |

## Remaining blockers by venue

- **Main-conference main-track:** not ready. The fully nested answer-product gain is positive but not significant; official sentence-level support metrics, multi-reader evidence, and a genuine federated evaluation are absent.
- **Findings/COLING-style positioning:** defensible after title/claim contraction. The paper can claim a statistically significant title-level evidence gain under a fully nested protocol, with neutral and statistically uncertain answer/product changes.
- **Artifact release:** usable, but the exact Hugging Face model and tokenizer revisions should be pinned before archival release.

## Non-negotiable wording rules

The submission must not use “official joint F1,” “privacy-preserving,” “secure,” “federated system improvement,” “generalizes to 2Wiki,” “reader-robust,” “SOTA,” or “outperforms SetR/RankRAG.” Any raw legacy field names must be accompanied by their custom metric definitions.
