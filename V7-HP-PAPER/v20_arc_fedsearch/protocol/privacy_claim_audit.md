# Privacy Claim Audit

| Claim | Allowed? | Rationale |
|---|---:|---|
| Data-local federated search | Yes | Raw corpus and full index remain client-local. |
| Communication-constrained retrieval | Yes | Client contacts and returned documents are explicitly budgeted. |
| Privacy-preserving retrieval | No | No DP, secure aggregation, TEE, HE, or PIR is implemented. |
| Private query-log training | Not yet | Train-only query logs may be used locally only after an explicit training contract. |
| Leakage-free final evaluation | Required | Final labels and IDs remain sealed until configuration freeze. |
