# Review Response: Strengths and Remaining Weaknesses

We thank the reviewers for identifying a real interpretation problem: the earlier presentation did not always separate resolved evidence gaps from persistent scientific limitations. The revision keeps all frozen outcomes, including failed tests, and reorganizes the paper around a bounded quality-risk-cost claim.

## Strengths retained without self-praise

- **Independent non-inferiority test:** Lite is frozen before the 3,405-query revision holdout and fails the 0.002 margin.
- **Budget-controlled comparison:** RECOMP, source-order truncation, the frozen baseline, and Full use the same FLAN reader and an approximately matched 660-token condition.
- **Policy-conditional decomposition:** population effects are separated from selected-query means, wins, losses, ties, drop rates, and exact fallbacks.
- **Fully nested no-leak protocol:** outer-test outcomes do not select generator modules, selector thresholds, or coverage.
- **Explicit failures:** Lite non-inferiority and 2Wiki transfer/calibration remain failures in the paper.

## 1. Missing latency

**Concern.** The method's cost could not be judged.  
**Agreement.** This was accurate for an earlier draft.  
**Evidence now available.** Same A100, batch size one, 50 warmup and 500 measured queries: Frozen Top-5 140.88 ms/query, Full 213.48, Lite 143.97, and RECOMP-660 169.64; one final reader call and 100% frozen-context match.  
**Paper location.** Section 7 and Appendix F.  
**Boundary.** This resolves the missing-measurement concern but does not make Full efficient; Full is 1.52x the baseline.

## 2. Complexity relative to gain

**Concern.** The architecture is elaborate relative to sub-point population gains.  
**Agreement.** Yes. The trade-off must be explicit.  
**Evidence.** Joint F1 changes by +0.0064 and +0.0080 on the two holdouts; Full adds 72.60 ms/query. Lite reduces cost but fails non-inferiority.  
**Paper location.** Abstract, Sections 5, 7, 10, and Conclusion.  
**Boundary.** We claim a bounded trade-off, not unqualified practical value.

## 3. External transfer

**Concern.** Same-source confirmation does not establish generalization.  
**Agreement.** Yes.  
**Evidence.** 2Wiki Answer/SP/Joint p-values are .1116/.6928/.3296; few-shot answer-drop 5.10% misses the 4% target.  
**Paper location.** Section 8 and Limitation 4.  
**Boundary.** The result is a failed transfer diagnostic, not external validation.

## 4. Candidate-pool scale

**Concern.** Pair construction may not scale to broader retrieval pools.  
**Agreement.** Yes.  
**Evidence.** The official pool is approximately ten documents; the frozen method scores ten pairs/query after pruning. Only one of 3,000 queries has at least twenty documents.  
**Paper location.** Section 7.1, Appendix I, and Limitation 5.  
**Boundary.** No corpus-scale or streaming claim is made.

## 5. Support replication across readers

**Concern.** The second reader may not independently validate the complete pipeline.  
**Agreement.** Correct.  
**Evidence.** FLAN and UnifiedQA have positive Answer deltas, but SP predictions are shared. Joint point estimates consequently share one component.  
**Paper location.** Section 9, Appendix H, and Limitation 6.  
**Boundary.** We call this directional answer-reader evidence, not independent SP or Joint replication.

## Final response boundary

The revision does not add tuning, reselect policies from holdout outcomes, or reinterpret non-significant results as success. Its contribution is a more accurate account of what the frozen evidence supports and what remains unresolved.
