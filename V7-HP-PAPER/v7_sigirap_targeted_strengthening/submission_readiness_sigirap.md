# SIGIR-AP Submission Readiness

oracle_diagnostic_complete: true
oracle_labeled_posthoc: true
opportunity_selection_gap_quantified: true
independent_reranker_complete: true
reranker_budget_fair: true
reranker_labeled_secondary: true
pair_pareto_complete: true
pair_pareto_labeled_exploratory: true
2wiki_structural_analysis_complete: true
subgroup_multiplicity_handled: true
no_holdout_retuning: true
no_primary_result_changed: true
paper_within_9_pages: true
main_story_clear: true
claims_safe: true
anonymous: true

final_grade: sigirap_ready

## Basis

- Main draft word count: 4667; it remains within the content budget of the prior 9-page manuscript. Final venue-template typesetting should still be rerun.
- Direct CrossEncoder latency status: `complete`.
- Frozen primary deltas remain +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080.

recommended_title: Opportunity-Aware Context Construction with Answer-Preserving Selection for Multi-Hop QA

one_sentence_claim: Under one frozen bounded pool and reader, Full provides modest replicated gains and higher Answer F1 than independent CE-Top5, while the secondary baseline recovers more SP/Joint gain and a retrospective frozen-action oracle reveals substantial selector regret.

main_new_evidence: The action-set diagnostic separates no-opportunity from selection misses on all 7,405 holdout queries, and the matched CrossEncoder baseline directly tests independent relevance under the same budget.

strongest_baseline_result: `ce_score_order` reaches Joint F1 0.3420/0.3405 versus Full 0.3356/0.3280, but Answer F1 is lower than Full by 0.0193/0.0181.

oracle_opportunity_result: Training-positive opportunity covers 22.8%/22.5% of the two holdouts; answer-preserving outcome-aware oracle Joint F1 is 0.4397/0.4251.

selector_regret_result: Aggregate policy/oracle gain ratios are 5.8%/7.6%, and mean Joint regret is 0.1041/0.0971; these are retrospective diagnostics.

latency_frontier_result: Pair pruning from ten to three evaluations preserves development Full quality in the frozen action replay but changes estimated total latency only from 213.48 to 212.04 ms/query; pair scoring is not the principal cost.

2wiki_boundary_result: No official 2Wiki reasoning-type subgroup survives BH-FDR correction; the taxonomy does not explain aggregate transfer uncertainty.

remaining_rejection_risk: The strongest risk is novelty positioning: independent CE ranking recovers or exceeds Full's Joint gain, population effects are small, Full costs 1.52x baseline, and all primary evidence uses an approximately ten-document Hotpot pool. The paper should be submitted as a transparent quality-risk analysis, not a universal pair-reranking win.

estimated_sigirap_probability_after_revision: 0.52 (subjective range 0.40-0.62)

do_not_run_second_retriever: true
do_not_retune_2wiki: true
