## 10. Limitations and Ethical Considerations

The population effects are below one and two F1 points, respectively, on the two same-source holdouts. Selected-query means are conditioned on the policy's choices and cannot be extrapolated to arbitrary or generally improvable queries. Most interventions tie the baseline, and 7.75-7.83% lower Answer F1 among selected interventions.

Both confirmatory sets come from HotpotQA validation. The 2Wiki experiment is non-significant and violates the answer-drop target even after few-shot calibration, so cross-dataset safety remains unresolved. Calibration also requires labeled target-train reader outcomes and is not zero-shot behavior.

Full adds measured online latency relative to Frozen Top-5. Lite reduces this overhead but fails the independent quality rule. Historical offline GPU-hour totals are unavailable. The benchmark begins after retrieval and does not include corpus indexing, network transfer, or retriever execution.

The bounded pool normally contains about ten distractor documents. Larger corpus-scale pools and changing indexes are not evaluated. The fixed reader and support predictor can have entity-, language-, or question-type-specific errors. A selector lowers average risk but offers no correctness guarantee. The method rearranges supplied passages and does not synthesize evidence; this helps auditability but cannot recover facts absent from the pool.
