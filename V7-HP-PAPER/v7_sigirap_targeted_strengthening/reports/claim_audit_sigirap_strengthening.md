# Claim Audit: SIGIR-AP Strengthening

| File | Section | Sentence / scan target | Evidence | Risk | Replacement |
| --- | --- | --- | --- | --- | --- |
| all audited files | global scan | `oracle upper bound proves` not found | n/a | None | n/a |
| all audited files | global scan | `policy lower bound` not found | n/a | None | n/a |
| all audited files | global scan | `captures X% of all possible gain` not found | n/a | None | n/a |
| all audited files | global scan | `generalizes to bridge questions` not found | n/a | None | n/a |
| all audited files | global scan | `solves transfer` not found | n/a | None | n/a |
| all audited files | global scan | `efficient variant` not found | n/a | None | n/a |
| all audited files | global scan | `Pareto-optimal deployment` not found | n/a | None | n/a |
| all audited files | global scan | `outperforms RankRAG` not found | n/a | None | n/a |
| all audited files | global scan | `outperforms RECOMP` not found | n/a | None | n/a |
| all audited files | global scan | `stronger than all rerankers` not found | n/a | None | n/a |
| all audited files | global scan | `real-world gain` not found | n/a | None | n/a |
| paper_sigirap_strengthened_9page.md:132 | scan | The selected subset has descriptive Answer/SP/Joint means of +0.0340/+0.0219/+0.0250 and +0.0447/+0.0237/+0.0309. These numbers condition on policy-selected interventions; they are not causal effects, oracle opportunity, or expected gains for arbitrary queries. Most interventions tie the baseline. Answer F1 decreases on 7.75% and 7.83% of selections; Joint F1 decreases on 14.86% and 14.19%. Exact wins, losses, ties, medians, and quartiles are in the supplement. | Explicitly negated in the sentence | None | n/a |
| paper_sigirap_strengthened_supplement.md:174 | scan | The 3,000 selected-policy means (+0.0340/+0.0219/+0.0250) and 3,405 selected-policy means (+0.0447/+0.0237/+0.0309) are descriptive conditional summaries. They are not causal effects, oracle opportunity, or expected gains for arbitrary queries. RECOMP-660 is a budget-controlled context-construction comparison only. UnifiedQA changes the answer reader but shares contexts and the support predictor. All candidate-outcome labels are offline and absent at inference. | Explicitly negated in the sentence | None | n/a |
| all audited files | global scan | `safe selection` not found | n/a | None | n/a |
| all audited files | global scan | `guaranteed preservation` not found | n/a | None | n/a |
| all audited files | global scan | `independent confirmation` not found | n/a | None | n/a |
| all audited files | global scan | `confirmatory new baseline` not found | n/a | None | n/a |
| all audited files | global scan | `SOTA` not found | n/a | None | n/a |

## Positive boundary checks

- Oracle is consistently called outcome-aware, retrospective, post-hoc, and non-deployable.
- CrossEncoder-Top5 is called a post-hoc secondary baseline, not a confirmatory comparison.
- The manuscript explicitly states that independent relevance recovers much of the SP/Joint gain.
- 2Wiki has no surviving FDR subgroup and no transfer-success claim.
- Full remains a quality-risk trade-off point rather than a universal winner.
