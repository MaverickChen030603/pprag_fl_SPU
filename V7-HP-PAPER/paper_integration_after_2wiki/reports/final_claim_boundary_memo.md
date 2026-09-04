# Final Claim Boundary Memo

## Claims We Can Make

1. HotpotQA v2.3 significantly improves joint_f1, support_recall@5, and sp_f1 under strict no-leak cross-fitting.
2. HotpotQA v2.3 preserves answer_f1 with a small non-significant positive delta.
3. Answer-neutral action selection helps bridge routing-side support gains and reader-side joint gains.
4. 2Wiki verifies that the adapter and reader-backed evaluation pipeline transfer to another multi-hop dataset.
5. 2Wiki reveals that cross-dataset selector generalization is limited by candidate exposure, feature detectability, and safety calibration.

## Claims We Cannot Make

1. answer_f1 significantly improves.
2. v2.3 selector generalizes successfully to 2Wiki.
3. 2Wiki validates the method as a main external result.
4. the selector reaches oracle upper bound.
5. no-leak selector can reliably identify all positive actions across datasets.

## Required Wording Discipline

Use "answer-preserving" for the HotpotQA answer-F1 result. Use "external diagnostic" or "limitation" for 2Wiki. Use "oracle diagnostic" only for upper-bound rows and never for inference-time method claims.
