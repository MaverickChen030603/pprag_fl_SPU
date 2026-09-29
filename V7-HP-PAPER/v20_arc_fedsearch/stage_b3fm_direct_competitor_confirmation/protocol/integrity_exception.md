# B3F-M Integrity Exception: Empty Threshold Context Support Prediction

Date: 2026-09-29

The frozen `ragroute_original_threshold` rule permits zero selected clients when
no mean RAGRoute probability exceeds 0.5. The reused C1/C2 Reader-support
pipeline has no defined behavior for an empty `reader_context_docs` list and
stopped when its support feature scaler received zero instances.

This is not a routing, model, retriever, budget, Reader-prompt, label, or data
integrity failure. However, the B3F-M pre-registration requires that the
support predictor and fallback behavior be frozen before blind execution. An
empty-context fallback was not explicitly specified. Therefore the blind run
is stopped rather than silently changing the support prediction mechanism.

Partial, non-frozen prediction files contain 24 rows per Reader and must not
be used. No checksum manifest, label unseal record, metric, or statistic has
been created. Continuing requires an explicit decision to either:

1. Amend the pre-registration and restart the entire B3F-M blind pipeline
   with the declared deterministic fallback `predicted_support=[]` for empty
   contexts; or
2. Remove `RAGRoute-Threshold` from B3F-M and keep it only as the separately
   completed B3F low-cost reference.

The first option changes the pre-registered support-prediction contract; the
second preserves the original B3F-M primary M2-vs-Fixed3 comparison but means
the threshold secondary reference is not part of this fresh confirmation.
