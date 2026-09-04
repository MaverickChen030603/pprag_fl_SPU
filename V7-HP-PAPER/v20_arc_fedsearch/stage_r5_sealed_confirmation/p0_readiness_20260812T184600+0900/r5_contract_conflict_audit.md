# R5 Contract Conflict Audit

## Status

**B — `R5_BLOCKED_CONTRACT_CONFLICT`**

## Authoritative Recommendation Extraction

| Field | Value |
|---|---|
| candidate_final_split | UNSPECIFIED |
| dataset_scope | UNSPECIFIED |
| sample_size | UNSPECIFIED |
| primary_method | UNSPECIFIED |
| primary_baseline | UNSPECIFIED |
| primary_metric | UNSPECIFIED |
| reader_scope | UNSPECIFIED |
| secondary_metrics | UNSPECIFIED |
| bootstrap_fdr_rule | UNSPECIFIED |
| go_no_go_gate | UNSPECIFIED |
| one_shot_execution_conditions | UNSPECIFIED (only states that final test remains sealed) |

## Conflicts

1. The recommendation inherits `probe_route_end_to_end_confirmed`, while the later authoritative provenance seal grades R4 `R4_VALID_EXPLORATORY_ONLY` and explicitly sets `r5_preregistration_ready=false`.
2. The recommendation does not specify a candidate split, datasets, N, primary comparison, Reader role, multiplicity rule, mechanical gate, or one-shot authorization conditions.
3. A later R5 preregistration was written at `2026-08-12T18:13:52+09:00` and invents three datasets, N=300, dual readers, and a statistics family not present in the recommendation.
4. The later contract does not define exact hash-selection seed/rule, a unique primary Reader versus robustness Reader, context Top-K/Bc/document budget/merge hashes, or success/inconclusive/harm/failure gates.
5. Its `pass_for_unlabeled_phases` audit provides no evidence of P0 completion or human approval and predates the final R4 provenance seal.
6. The R5 pipeline started at `2026-08-12T18:21:58+09:00`, before this P0 audit and before a human-approved draft. At P0 start, retrieval was active; this audit did not start, stop, resume, or alter it.
7. The conservative power plan requires approximately 4110 per primary cell, whereas the later unapproved contract selected 300.

## Minimal Human Decisions Required

- Decide whether R4 remains exploratory-only and whether a new confirmation is still required.
- Name a genuinely new candidate split/data source and approve a query-ID-only selection policy.
- Freeze one unique primary dataset/Reader/comparison or explicitly define a multiplicity family.
- Freeze N, alpha, power target, context/retrieval/Reader hashes, and mechanical success/harm/inconclusive gates.
- Define a human authorization record and post-prediction sealed-label access procedure.
- Decide how to treat the already-started `stage_r5_final_test/run_20260812` execution; its selected IDs cannot simply be rerun as a fresh confirmation.

No preregistration draft or synthetic runner dry-run is generated under status B.
