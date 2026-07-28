# V17 Federated Evidence Dispersion Audit

Gold evidence is used only for this offline audit. It is unavailable to partitioning, routing, retrieval, and action generation.

| Dataset | Partition | N | Cross-client | Two-client | Three-plus | One-client | Mean entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | dirichlet_a0p1 | 1000 | 0.869 | 0.677 | 0.192 | 0.131 | 0.707 |
| 2wikimultihopqa | dirichlet_a0p3 | 1000 | 0.901 | 0.706 | 0.195 | 0.099 | 0.733 |
| 2wikimultihopqa | dirichlet_a1p0 | 1000 | 0.962 | 0.762 | 0.200 | 0.038 | 0.784 |
| 2wikimultihopqa | entity_community | 1000 | 0.712 | 0.583 | 0.129 | 0.288 | 0.545 |
| 2wikimultihopqa | random_control | 1000 | 0.961 | 0.762 | 0.199 | 0.039 | 0.784 |
| 2wikimultihopqa | topic_silo | 1000 | 0.677 | 0.529 | 0.148 | 0.323 | 0.527 |
| hotpotqa | dirichlet_a0p1 | 1000 | 0.773 | 0.773 | 0.000 | 0.227 | 0.536 |
| hotpotqa | dirichlet_a0p3 | 1000 | 0.892 | 0.892 | 0.000 | 0.108 | 0.618 |
| hotpotqa | dirichlet_a1p0 | 1000 | 0.919 | 0.919 | 0.000 | 0.081 | 0.637 |
| hotpotqa | entity_community | 1000 | 0.728 | 0.728 | 0.000 | 0.272 | 0.505 |
| hotpotqa | random_control | 1000 | 0.969 | 0.969 | 0.000 | 0.031 | 0.672 |
| hotpotqa | topic_silo | 1000 | 0.443 | 0.443 | 0.000 | 0.557 | 0.307 |
| musique | dirichlet_a0p1 | 1000 | 0.905 | 0.689 | 0.216 | 0.095 | 0.721 |
| musique | dirichlet_a0p3 | 1000 | 0.942 | 0.705 | 0.237 | 0.058 | 0.757 |
| musique | dirichlet_a1p0 | 1000 | 0.946 | 0.717 | 0.229 | 0.054 | 0.757 |
| musique | entity_community | 1000 | 0.587 | 0.479 | 0.108 | 0.413 | 0.444 |
| musique | random_control | 1000 | 0.967 | 0.722 | 0.245 | 0.033 | 0.778 |
| musique | topic_silo | 1000 | 0.782 | 0.632 | 0.150 | 0.218 | 0.600 |
