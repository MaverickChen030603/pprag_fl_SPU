# Submission Readiness Report

## Machine-readable decision

```yaml
fully_nested_protocol_complete: true
official_metric_complete: false
data_provenance_complete: true
federated_claim_defensible: false
citation_complete: true
reproducibility_complete: false
recommended_venue_tier: findings_or_coling_ready
```

## Decision

**Recommended tier: `findings_or_coling_ready`.**

The package now supports a protocol-valid and metric-honest paper about reader-safe context action selection. The fully nested 1,000-query rerun is complete, source sampling is exactly reconstructed, citations are verified, and custom metrics are named correctly. The main empirical claim is statistically supported for title-level evidence recall/F1.

It is not main-conference-ready. The answer-F1 and answer-title-product gains are not significant, official HotpotQA support/joint predictions do not exist, the organizer is centralized over synthetic clients, and only one reader completed. Reproducibility is also marked false because the exact historical model/tokenizer revisions and raw-reader runtime were not logged, despite the otherwise complete package.

## Recommended title

**Reader-Safe Context Action Selection for Multi-Hop Question Answering**

## Recommended claim

> Under fully nested query-level cross-fitting on a reproducibly sampled HotpotQA-derived 1,000-query evaluation, a bounded answer-safety-gated organizer significantly improves title-level support recall and title-level support F1. Average answer F1 and the answer-title-support product remain statistically uncertain, and the dominant limitation is the absence of a positive main-eligible action for 797 queries.

## Why this tier is defensible

- Strict outer-fold leakage is repaired and audited.
- The title-level evidence gains remain positive and significant.
- The safety ablation supplies a coherent method result: removing safety reverses the answer mean.
- The negative boundaries are informative and explicit: no official joint metric, no 2Wiki transfer, no reader robustness, and no end-to-end federated claim.
- The method is modest, but the protocol and opportunity analysis form a complete Findings/COLING-style contribution.

## Mandatory remaining actions before upload

1. Pin and record exact `google/flan-t5-large` model and tokenizer revisions, or state in the reproducibility checklist that the historical cache cannot be recovered.
2. Create a clean archival Git tag/commit containing the submission scripts and manifests.
3. Convert the Findings markdown into the target venue template and verify page limits, anonymization, ethics/reproducibility sections, and bibliography rendering.
4. Ensure every table uses “title-level support” and “answer-title-support product,” never bare legacy `sp_f1/joint_f1`.

## Optional strengthening actions

1. Add predicted support sentence IDs and official HotpotQA scoring.
2. Pre-specify and rerun a higher-coverage or expanded action generator on untouched queries.
3. Complete one additional reader.
4. Replace synthetic client assignment with a documented non-IID distributed evaluation only if Federated RAG returns to the title.
5. Evaluate candidate opportunity beyond the 1,000-query sample.

## File choice

- Submit from `paper_submission_findings_v2.md` after venue formatting.
- Use `paper_full_clean_v2.md` as the complete internal source.
- Keep `paper_appendix_v2.md` as the supplementary source.
- Do not submit `paper_submission_main_v2.md`; it is intentionally status-marked.
