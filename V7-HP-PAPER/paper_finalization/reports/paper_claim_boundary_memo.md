# Paper Claim Boundary Memo

## Claims We Can Make

1. The proposed selector improves joint_f1 significantly under strict no-leak cross-fitting.
2. It improves support_recall@5 and sp_f1 significantly.
3. It preserves answer_f1 with a small non-significant positive delta.
4. Positive-action recall improves substantially compared with v2.2.
5. The candidate pool still limits achievable gains.

## Claims We Should Not Make

1. It significantly improves answer_f1.
2. It solves reader sensitivity completely.
3. It reaches oracle upper bound.
4. It works for all multi-hop QA cases.
5. It proves support gain always improves answer generation.

## Recommended Contribution Phrase

Answer-neutral action selection for federated RAG routing.

Alternative: Bridging routing-side support gains and reader-side joint QA gains under no-leak constraints.
