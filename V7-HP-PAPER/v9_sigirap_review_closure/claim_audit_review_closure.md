# Claim Audit: Review Closure

| Phrase | File/section | Evidence | Risk | Required disposition |
| --- | --- | --- | --- | --- |
| `leak-free` | none | No positive assertion found | Prohibited absolute wording | Use leakage-controlled plus explicit audit |
| `independent holdout` | none | No positive assertion found | Could imply external replication | Use disjoint same-source evaluation |
| `external replication` | paper_sigirap_review_closed_9page.md | Occurrence reviewed in bounded/negated context | Unsupported | State same-source replication |
| `guarantee` | paper_sigirap_review_closed_9page.md, paper_sigirap_review_closed_supplement.md | Occurrence reviewed in bounded/negated context | Needs formal scope | Negate or attribute to conformal work |
| `certified` | paper_sigirap_review_closed_9page.md, paper_sigirap_review_closed_supplement.md | Occurrence reviewed in bounded/negated context | Needs formal scope | Use only for C-RAG citation or explicit negation |
| `conformal` | paper_sigirap_review_closed_9page.md, paper_sigirap_review_closed_supplement.md | Occurrence reviewed in bounded/negated context | Could conflate methods | State current method is not conformal |
| `robust reader` | none | No positive assertion found | Too broad | Use Answer-only directional evidence |
| `pair complementarity necessary` | none | No positive assertion found | No frozen holdout removal | Use diagnostic consistency only |
| `Full outperforms` | none | No positive assertion found | Universal comparison unsafe | Name metric and baseline |
| `Pareto-optimal` | none | No positive assertion found | Search-space universal claim | Use non-dominated among evaluated points |
| `scalable` | none | No positive assertion found | No large-pool evaluation | Bound to approximately ten documents |
| `causal` | paper_sigirap_review_closed_9page.md, paper_sigirap_review_closed_supplement.md | Occurrence reviewed in bounded/negated context | Observational diagnostic | Use retrospective/descriptive |
| `oracle achievable` | none | No positive assertion found | Oracle uses target outcomes | Call retrospective, not deployable |
| `deployment-ready` | none | No positive assertion found | Unsupported | Omit or explicitly negate |
| `federated` | paper_sigirap_review_closed_9page.md | Occurrence reviewed in bounded/negated context | Not evaluated in paper claim | Omit |
| `privacy` | paper_sigirap_review_closed_9page.md | Occurrence reviewed in bounded/negated context | Not evaluated | Explicitly state no privacy claim |

## Outcome

No primary claim depends on holdout retuning, fabricated removal variants, external-domain replication, finite-sample safety, universal CrossEncoder superiority, corpus-scale behavior, privacy, or federated deployment. Terms such as `guarantee`, `conformal`, and `certified` appear only in related-work distinctions or explicit negations.
