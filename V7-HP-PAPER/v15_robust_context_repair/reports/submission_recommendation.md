# V15 Submission Recommendation

Status: `needs_more_experiments`.

- **Recommended method name:** not frozen. Retain *Robust Risk-Calibrated
  Context Repair* as an internal working name until the final collision audit.
- **One-sentence method claim:** V15 searches complete bounded context repairs,
  predicts reader-specific official-metric deltas, and applies one independent
  risk-aware action or an exact fallback; the deployable gain is still under
  validation.
- **Main technical novelty:** the combination of real-pool complete-sequence
  repair, direct multi-objective/multi-reader utility, per-query risk gating,
  and exact fallback, with a cost-aware cascade planned as part of the method.
- **Main empirical contribution so far:** on a fresh 100-query HotpotQA pilot,
  the expanded action set contains reader-positive repairs for both frozen
  readers, with Joint oracle deltas of +0.0528 and +0.0474. These are explicitly
  upper bounds, not main results.
- **Did direct utility solve proxy mismatch?** Partially. Average direct Joint
  correlation improves over the old cheap proxy, and harm detection is useful
  for FLAN, but UnifiedQA Joint ranking remains negative and the unconstrained
  top action is not robust.
- **Did the per-query gate replace batch Top-B?** Implemented in code, but not
  yet calibrated or validated on the independent calibration/final splits.
- **Does multi-reader robustness hold?** Not yet as a confirmatory result. A
  beta=1 inference-safe diagnostic is positive on 22 held-out pilot queries
  (+0.0113 mean-reader Joint; +0.0022 minimum-reader mean; no observed harm),
  which justifies further scaling.
- **Did opportunity improve at Top-20?** Yes at retrieval and action-set levels
  in development pilots. Full pool-level/search-level decomposition is pending.
- **Did latency improve over old Full?** Not established. Repair generation is
  84.6 ms/query and the MLP forward pass is cheap, but the first cheap gate must
  invoke the expensive stage on 100% of the 22-query development pilot to retain
  90% opportunity recall. No end-to-end latency claim is allowed.
- **Is V15 dominated by CrossEncoder?** Unknown until quality, risk, and cost are
  evaluated under the same frozen pools/readers.
- **Largest remaining weakness:** the selector does not yet learn a consistent
  cross-reader ranking, and 2Wiki Top-20 complete-support recall is only 0.40 in
  the 50-query smoke.
- **ECIR readiness:** promising method prototype, not submission-ready.
- **ARR readiness:** not ready.
- **Recommended route:** expand reader-labelled training queries, freeze robust
  beta and gates on development/calibration, complete 2Wiki and cost analysis,
  then execute the untouched final splits. If cross-reader/cross-dataset signals
  fail Checkpoint 3, reposition as an ECIR analysis paper rather than a method
  breakthrough.
