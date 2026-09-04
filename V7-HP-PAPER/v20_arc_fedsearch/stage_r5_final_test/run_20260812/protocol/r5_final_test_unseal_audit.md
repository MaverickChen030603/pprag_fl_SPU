# R5-P0 Final-Test Unseal Audit

**Decision:** `pass_for_unlabeled_phases`; labels remain sealed.

- Frozen scientific commit: `13091c6deb6b3868705a49041e89f578f14b4e0e`.
- Audited 148 R2--R4 structured artifacts; final-query overlap is 0 for all datasets.
- V17 final inputs contain query/corpus fields only; sealed labels were checked by path, mode and SHA-256 only.
- Frozen sample: N=300/dataset, label-free hash selection retained in source order.
- This is a train-derived untouched held-out split, not an official hidden test.
- Authorization covers Phase 1 retrieval and Phase 2 unscored reader inference only.
