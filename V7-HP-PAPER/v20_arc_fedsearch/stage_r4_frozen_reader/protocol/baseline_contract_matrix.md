# Hotpot H0/C0 Baseline Contract Matrix

| Component | H0 inherited route (R4 primary) | C0 static Top-3 (cost-only) |
|---|---|---|
| Query IDs/sample | R3-T frozen Hotpot holdout N=300 | same R3-T holdout N=300 |
| Candidate generator | inherited topic origin-plus-centroid route | static P0 centroid profile |
| Selected clients | frozen `inherited_b3_routes.jsonl` | P0 static ranks 1--3 |
| Local depth/transmission | 10 local materialized, 5 per client, 15 total | identical |
| Global merge/pool | raw dense score, Top-10 | identical |
| Corpus/partition/index | V17 topic-silo canonical index and assignment | identical |
| R3 observed raw complete@10 | 0.4700 | 0.5100 |
| Role in R4 | **federated baseline** | communication/Pareto comparison only |

The difference is explained by route selection, not a detected implementation failure. R4 therefore uses H0 only as the Hotpot main federated baseline. C0 is not admitted to the reader comparison.
