# V19 Update-selection Literature and Name-Conflict Audit

**Audit date:** 2026-07-31
**Decision:** retain `RASU-FedRAG` only as an internal identifier until the
method passes Stage 2; do not use it as a title claim.

| Work family | Representative work | What it already establishes | V19 non-overlapping question |
|---|---|---|---|
| Federated retriever adaptation | [FedE4RAG](https://arxiv.org/abs/2504.19101) | Federated training of client RAG retrievers with privacy mechanisms and distillation. | Which *retriever parameter blocks* merit transmission under a fixed payload when judged by frozen reader outcomes? |
| Parametric FedRAG adapters | [FedMosaic](https://arxiv.org/abs/2602.05235) | Parametric document adapters, masked multi-document adapters, and relevance-aligned/nonconflicting selective adapter aggregation. | It is not valid to claim first adapter-based FedRAG or first selective aggregation. V19 is restricted to local retriever-update blocks and independent reader-probe credit. |
| Federated optimization | [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html), [FedProx](https://arxiv.org/abs/1812.06127), [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html) | Full-model aggregation, proximal stabilization, and control variates for heterogeneous clients. | Does their proxy-agnostic update handling preserve reader utility as well as reader-aligned block selection? |
| Communication compression | [FedPAQ](https://arxiv.org/abs/1909.13014), [Federated Dropout](https://arxiv.org/abs/2112.10663) | Quantization, periodic averaging, and subnet/model dropout reduce transmitted payload. | Is payload allocation guided by a reader-aligned utility more effective than random/subnetwork/proxy allocation at equal bytes? |
| Federated RAG routing | [RAGRoute](https://arxiv.org/abs/2507.12123) and [FeB4RAG](https://arxiv.org/abs/2411.14073) | Query/source routing and federated retrieval evaluation are distinct from training-update selection. | V19 freezes the primary source router and uses its routed pool only to observe retriever-to-reader transmission. |

## Claim guardrails

1. Do not claim the first selective FedRAG method, first source router, first
   personalized federated retriever, first adapter approach, or first
   non-conflicting aggregation.
2. Do not call a retrieval-only improvement a QA improvement. Reader outcomes
   are separately measured with two frozen readers.
3. `reader-aligned` means exact utility labels are computed on local probes that
   never take gradient updates. It does not mean the reader is tuned or queried
   at inference.
4. V19 may claim a method contribution only after the Stage 0--2 gates pass on
   the pre-registered data splits and equal-payload baselines.
