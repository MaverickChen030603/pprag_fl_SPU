# Final Claim Audit

| File | Section | Audited sentence / risky claim | Evidence | Risk | Replacement / final status |
| --- | --- | --- | --- | --- | --- |
| paper_full_clean_v4_final.md | 7 Development Behavior | "Joint F1 is positive but non-significant." | Delta +0.0064, p=0.0752 | Low after correction | Replaces the forbidden significant-joint claim. |
| paper_main_conference_v4_final.md | 7 Development Behavior | "Joint F1 rises by 0.0064, but its interval includes zero." | 95% CI [-0.0005,+0.0132] | Low after correction | Approved. |
| paper_anonymous_v4_final.md | Abstract | "Joint F1 shows a positive, non-significant trend." | p=0.0752 | Low after correction | Approved. |
| paper_full_clean_v4_final.md | 9 External Transfer | "The result does not establish statistically reliable transfer." | Answer/joint CIs include zero; SP flat | Low after correction | Replaces "generalizes to 2Wiki." |
| paper_main_conference_v4_final.md | 9 External Transfer | "The frozen pipeline preserves positive answer and joint point estimates, but all confidence intervals include zero." | p=0.1116/0.6928/0.3296 | Low after correction | Approved; no cross-dataset robustness phrase. |
| paper_anonymous_v4_final.md | 9 External Transfer | Same bounded external sentence as main paper. | One non-significant external sample | Low after correction | Section title is Generalization Boundary. |
| paper_main_conference_v4_final.md | 7 Frozen Holdout | "The support predictor is shared, so the second reader confirms answer/joint direction rather than independent support robustness." | One shared support predictor | Low after correction | Replaces multi-reader full-pipeline claim. |
| paper_appendix_v4_final.md | G Multi-Reader | "These are not independent support replications." | Support predictions are shared | Low after correction | Approved. |
| paper_main_conference_v4_final.md | 6 Baselines | "Official-code reproduction under a standardized reader adaptation." | Reader changed from FLAN-UL2 to FLAN-T5-Large | Low after correction | Replaces faithful end-to-end reproduction. |
| paper_appendix_v4_final.md | H RECOMP Fairness | "The large gap is confounded by the 7.35% token ratio." | 47.13 vs 668.18 context tokens | Medium residual risk | General superiority claim removed. |
| paper_main_conference_v4_final.md | 8 Generator Components | "MPNet, cross-encoder, missing-hop, and redundancy ablations are mixed." | Several removals improve raw coverage | Low after correction | Replaces all-components-positive claim. |
| paper_main_conference_v4_final.md | 7 Opportunity | "It passes ... three criteria, but misses overall coverage and efficiency." | Three of five criteria pass | Low after correction | Replaces all-gates-pass claim. |
| paper_anonymous_v4_final.md | Entire file | No state-of-the-art claim. | No comprehensive benchmark | Low | Machine scan passes. |
| paper_anonymous_v4_final.md | Entire file | No Federated RAG method identity or privacy claim. | No federated/privacy mechanism evaluated | Low | Machine scan passes. |
| paper_main_conference_v4_final.md | 6 Statistics | "No immutable pre-run hierarchy was found; we do not claim formal ordered testing." | Plan timestamp follows holdout result | Low after correction | Replaces ordered-confirmatory language. |
