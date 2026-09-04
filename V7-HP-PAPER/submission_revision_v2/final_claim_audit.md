# Final Claim Audit

Audit target: `paper_full_clean_v2.md`, `paper_submission_findings_v2.md`, `paper_submission_main_v2.md`, and `paper_appendix_v2.md`.

## Findings

| File / location | Claim or trigger | Evidence sufficient? | Risk | Disposition |
|---|---|---|---|---|
| Full, Abstract and Sec. 4.4 | “fully nested” | Yes | Low | Supported by five 800/200 outer splits, inner-OOF nuisance predictions, frozen outer-test inference, zero overlap, and `outer_test_outcome_used=false`. Keep. |
| Findings, Abstract and Method | “fully nested five-fold query-level cross-fitting” | Yes | Low | Same evidence. Keep. |
| Full, Abstract/Results | Title recall +0.0120, p=0.007; title F1 +0.0150, p=0.018 | Yes | Low | Matches `nested_significance_report.json`. Keep custom title qualifiers. |
| Full and Findings, Results | Answer +0.0028 and product +0.0079 | Yes, as descriptive means | Medium | Both are explicitly non-significant with CIs crossing zero. Never describe as established gains. |
| Full, Metric setup; Appendix A | Legacy `sp_f1` / `joint_f1` names | Yes, as artifact mapping | Medium | Both occurrences immediately state that fields are custom title F1/product, not official metrics. Keep only for traceability. |
| Full/Findings/Main, official HotpotQA references | Official metrics unavailable or future work | Yes | Low | All uses are negated, limitation, or required future work. No official benchmark improvement claim remains. |
| Full, Introduction/Related Work/Limitations | Distributed or federated motivation | Yes, as motivation | Medium | Centralized text and synthetic IDs are stated. No end-to-end Federated RAG system result is claimed. Route B is consistent. |
| Appendix K | No secure aggregation/privacy/non-IID evaluation | Yes | Low | Correct negative boundary. No privacy-preserving or secure-system claim. |
| Full/Findings/Appendix, 2Wiki | Transfer does not establish generalization | Yes | Low | Every occurrence is a failure boundary. No cross-dataset claim. |
| Findings, Limitations | “no reader robustness” | Yes | Low | Explicitly negated; one reader only. |
| Full/Findings, candidate opportunity | 797 queries have no positive main-eligible action | Yes | Low | Matches `action_scope_statistics.json`. Replaces the v1 main-text misuse of all-template 778. |
| Appendix E | 778 all-template and 797 main-eligible | Yes | Low | Both scopes are named and reconciled. |
| Main version, first line | `NOT READY FOR MAIN-CONFERENCE SUBMISSION` | Yes | Low | Required because official metrics, federated framing, multi-reader evidence, and significant answer effect are absent. |

## Trigger scan outcome

- `strict no-leak`: absent. The paper uses the more auditable “fully nested” description instead.
- `official HotpotQA`: appears only in negative/future-work statements.
- `joint F1`: no unqualified metric claim; legacy `joint_f1` appears only in definition audits.
- `Federated RAG system`: no positive evaluated-system claim.
- `privacy-preserving` / `secure`: no positive claim.
- `generalizes to 2Wiki`: absent; failure boundary stated.
- `reader-robust`: no positive claim.
- `outperforms SetR`, `SOTA`, `all context actions`: absent.

## Required wording preserved

Use:

- “title-level support recall@5”;
- “title-level support F1”;
- “answer-title-support product”;
- “HotpotQA-derived 1,000-query evaluation”;
- “fully nested query-level cross-fitting”;
- “centralized organizer over synthetic client identities.”

Avoid:

- “official joint F1”;
- “HotpotQA benchmark improvement” without the title-level qualifier;
- “federated privacy” or “communication-efficient organizer”;
- “generalizes across datasets/readers”;
- “all context actions” or “optimal coverage.”

## Audit decision

The Findings and full-v2 manuscripts are internally claim-consistent. The main-track version is correctly status-blocked. No P0 claim-language violation remains.
