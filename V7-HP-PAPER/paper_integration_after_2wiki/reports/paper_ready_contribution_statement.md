# Paper-Ready Contribution Statement

This paper introduces answer-neutral action selection for federated RAG routing. On HotpotQA, the final v2.3 selector converts routing-side support gains into significant reader-side improvements in joint_f1, support_recall@5, and sp_f1 under strict no-leak query-level cross-fitting, while preserving answer_f1 with a small non-significant positive delta.

The method should be positioned as a support-to-joint bridge rather than as an answer-F1 optimizer. The key empirical contribution is that answer-neutral filtering prevents routing improvements from damaging answer anchors, enabling support-side retrieval gains to survive downstream reader evaluation.

As an external diagnostic, 2WikiMultiHopQA shows that the adapter and reader-backed evaluation pipeline transfer to another multi-hop dataset, but selector-level generalization remains limited against a strong BM25 baseline. This limitation motivates future work on candidate generation beyond BM25 and dataset-robust safety calibration.
