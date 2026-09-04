# Submission Readiness: Review Closed

review_facts_corrected: true
leakage_wording_bounded: true
same_source_wording_correct: true
reproducibility_details_complete: true
crossencoder_dual_role_clear: true
end_to_end_ablations_valid_or_deferred: true
support_threshold_sensitivity_complete: true
reader_claim_bounded: true
latency_p95_restored: true
conformal_scope_clear: true
oracle_diagnostic_status_clear: true
pool_scaling_boundary_clear: true
joint_ci_visible: true
no_holdout_retuning: true
no_primary_result_changed: true
within_9_pages: true
anonymous: true
claims_safe: true

final_status: sigirap_ready_with_review_risk

recommended_title: Opportunity-Aware Context Construction with Risk-Controlled Selection for Multi-Hop QA

one_sentence_claim: On two disjoint frozen same-source HotpotQA evaluations, bounded opportunity-aware actions plus an empirically risk-controlled fallback improve Answer, SP, and Joint over Frozen Top-5, while a shared-checkpoint CrossEncoder defines a stronger evidence-oriented operating point.

main_method_contribution: Bounded pair-complementary and anchor-preserving context actions with nested preservation/utility selection and exact fallback.

main_analysis_contribution: Separation of candidate availability, selector realization, observed intervention risk, and answer-evidence-latency operating points under a fully nested leakage-controlled protocol.

strongest_review_resolved: Reproducibility, CrossEncoder role isolation, support-threshold sensitivity, latency variance, and risk/conformal scope are now explicit and artifact-backed.

largest_unresolved_weakness: No clean pre-inspection frozen end-to-end removal checkpoints exist for pair, chain, or CrossEncoder features; component attribution therefore remains development-diagnostic.

crossencoder_interpretation: Protocol-matched shared-checkpoint relevance-only baseline; higher SP/Joint and lower latency, but lower Answer than Full.

oracle_interpretation: Retrospective bounded-action diagnostic that separates availability and selector regret; not a baseline, guarantee, or deployable policy.

reader_robustness_boundary: FLAN and UnifiedQA show positive Answer direction, but the support predictor is shared, so SP/Joint are not independent reader replications.

scaling_boundary: Natural candidate pools are approximately ten documents; no common 20/50 pool or corpus-scale claim is available.

estimated_sigirap_probability: 0.52

recommended_submission_decision: Submit after official-template page compilation and a final citation-key check; retain `sigirap_ready_with_review_risk`, not Strong Accept.
