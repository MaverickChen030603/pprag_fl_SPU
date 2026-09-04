# Review Accuracy and Action Audit

| Review concern | Classification | Evidence audit | Action in v9 |
| --- | --- | --- | --- |
| "Two independent frozen holdouts" | Valid and actionable | Samples are disjoint but both come from HotpotQA distractor validation | Replaced with "two disjoint frozen same-source evaluations" throughout |
| "Leak-free protocol" | Partially resolved | Nested split and no-leak artifacts exist, but absolute leakage freedom is too strong | Uses "fully nested, leakage-controlled protocol with an explicit no-leak audit" |
| Latency variance missing | Already resolved in artifacts | Mean, median, P95, component means, calls, and memory were frozen | Restored mean/P95 to main and complete breakdown to supplement |
| Oracle should also appear on development | Already resolved in artifacts | Nested development decomposition is 708/213/79; AP-oracle Joint is .4404 | Added development row and bounded retrospective interpretation |
| End-to-end component removals are needed | Valid and actionable, but unavailable as clean confirmation | No compatible removal checkpoint was frozen before holdout inspection | Retains development opportunity diagnostics and explicitly defers holdout removals |
| CrossEncoder role is confounded | Valid and actionable | Same relevance checkpoint is used under different selection contracts | Added shared-checkpoint fairness audit and protocol-matched terminology |
| Support score may depend on threshold 0.7 | Valid and actionable | Frozen probabilities permit a no-training fixed-grid audit | Added 0.5/0.6/0.7/0.8 sensitivity; no direction flips and no threshold change |
| Risk-controlled wording implies guarantees | Valid and actionable | Current gate is empirical, not conformal or PAC-style | Added formal distinction and removed guarantee-like wording |
| Need natural 20/50 pool experiment | Valid motivation but out of scope for this benchmark | 2,973/3,000 have at least ten, one at least twenty, none fifty | States bounded-pool limit; does not fabricate corpus-scale stress test |
| Full should generally beat CrossEncoder | Incorrect or overstated | CrossEncoder has higher SP/Joint and lower latency; Full has higher Answer | Frames distinct operating points and avoids universal superiority |
| UnifiedQA independently replicates Joint | Incorrect or overstated | Answer reader changes, support predictor is shared | Reports Answer-only directional evidence |
| Conformal gate should be added now | Valid but out of scope | Would need separate assumptions, calibration and frozen validation | Related-work/future-work addition only |
