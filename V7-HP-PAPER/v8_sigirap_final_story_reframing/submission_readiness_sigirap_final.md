# Final SIGIR-AP Submission Readiness

title_risk_bounded: true
abstract_not_failure_list: true
candidate_and_selector_bottlenecks_separated: true
crossencoder_tradeoff_clear: true
crossencoder_answer_loss_visible: true
no_metric_hierarchy_overclaim: true
oracle_compressed_in_main: true
oracle_posthoc_label_clear: true
pair_pruning_demoted: true
2wiki_claim_bounded: true
no_federated_scope_creep: true
no_edge_claim: true
cost_for_all_queries_clear: true
main_results_frozen: true
within_9_pages: true
anonymous: true
claims_safe: true

final_grade: sigirap_ready_with_review_risk

## Evidence

- Main-paper word count: 4440, below the previous strengthened draft's 4,667-word content budget. Final venue-template typesetting remains required.
- Abstract body: 195 words, within the requested 170-210 range.
- Frozen primary deltas remain +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080.
- CrossEncoder Answer/SP/Joint, latency, and paired Full contrasts are unchanged.
- Oracle absolute scores and gain ratios are removed from Abstract and Conclusion and retained in the supplement.
- Pair pruning is supplement-only apart from one cost-diagnosis sentence.
- The claim audit contains 0 unresolved medium-risk positive assertions.

recommended_title: Opportunity-Aware Context Construction with Risk-Controlled Selection for Multi-Hop QA

one_sentence_claim: Under a frozen bounded HotpotQA pool, Full offers a modest Answer-oriented selective operating point, while independent CrossEncoder reranking offers stronger SP/Joint at lower latency and retrospective diagnostics reveal separate candidate-availability and selector-regret limitations.

main_method_contribution: Bounded pair-complementary, anchor-preserving context actions with a fully nested risk-controlled selector and exact fallback.

main_analysis_contribution: A leak-controlled decomposition of candidate availability, selector regret, intervention risk, and answer-evidence-latency trade-offs under matched reranking.

crossencoder_tradeoff: CrossEncoder-Top5 reaches Joint F1 0.3420/0.3405 versus Full 0.3356/0.3280 at 149.90 versus 213.48 ms/query, while its Answer F1 is 0.0193/0.0181 below Full and 0.0105/0.0066 below baseline.

oracle_interpretation: The outcome-aware frozen-action oracle reveals substantial selector regret, but it is retrospective, non-deployable, and does not remove the separate no-positive-action limitation.

main_rejection_risk: A strong reviewer may view CrossEncoder as sufficient for SP/Joint, regard the method novelty as incremental, and judge Full's small same-source gains insufficient for its 1.52x baseline latency.

estimated_acceptance_probability: 0.46 (subjective range 0.34-0.58)

recommended_submission_decision: Submit to SIGIR-AP with the final multi-objective framing; do not restore method-superiority, federated, edge, or cross-domain claims.
