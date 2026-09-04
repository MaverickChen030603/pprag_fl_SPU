# Final Claim Audit

| Proposed claim | Decision | Evidence / correction |
| --- | --- | --- |
| Semantic generation improves action density and query opportunity over V3 | Allowed | Density 9.43% to 14.71%; coverage 23.4% to 29.2% |
| The candidate-opportunity bottleneck is solved | Forbidden | Overall coverage remains 29.2%; two of five criteria fail |
| All opportunity criteria pass | Forbidden | Only B, C, and D pass |
| Fully nested V4 improves development answer and support F1 | Allowed | +0.0133, p=0.0176; +0.0053, p=0.0372 |
| Development joint F1 improves significantly | Forbidden | +0.0064, p=0.0752 |
| Frozen holdout reproduces answer/support/joint gains | Allowed | FLAN +0.0088/+0.0056/+0.0064 with p=0.0096/0.0004/0.0004 |
| The holdout is cross-dataset | Forbidden | It is a disjoint same-source HotpotQA sample |
| Answer and joint directions are consistent across two readers | Allowed | FLAN and UnifiedQA are direction-consistent |
| Support prediction independently replicates across readers | Forbidden | One frozen support predictor is shared |
| Zero-shot 2Wiki results establish broad generalization | Forbidden | Answer/joint are positive but non-significant; support is flat |
| Frozen external transfer is non-degrading in answer/joint point estimates | Allowed with boundary | +0.0086 answer F1 and +0.0033 joint F1; both CIs cross zero |
| V4 outperforms the reproduced RECOMP compressor under the standardized reader | Allowed | Joint F1 0.3305 versus 0.2084 |
| Exact end-to-end RECOMP reproduction | Forbidden | The reader is adapted to FLAN-T5-Large |
| Every semantic submodule is independently necessary | Forbidden | Component ablations are mixed; no-document-model improves raw opportunity |
| Pair complementarity and two-document actions are important opportunity components | Allowed | Removing them lowers coverage to 27.7% and 25.1% |
| The work is a Federated RAG or privacy-preserving system | Forbidden | Current experiments evaluate centralized reader-side context actions |
| Opportunity criteria were preregistered | Forbidden | No public immutable registration record was found |
| Opportunity criteria were pre-specified | Allowed | Versioned local artifact and execution order support the weaker wording |
| State of the art | Forbidden | One close baseline is reproduced; no comprehensive leaderboard claim |
