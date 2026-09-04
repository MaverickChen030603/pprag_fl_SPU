# ARS Reviewer Panel on the Polished Draft

## Domain reviewer

**Assessment:** The revised story is clearer and more consequential. Replication across two frozen holdouts now appears before diagnostic caveats, and the CrossEncoder comparison is integrated as a multi-objective finding rather than treated as an embarrassing add-on.

**Score:** 7/10, accept.  
**Main concern:** Pair complementarity remains only partially distinguished from strong relevance reranking.

## Methodology reviewer

**Assessment:** The statistical and leakage boundaries remain intact. Absolute scores, confidence intervals, post-hoc labels, Answer-drop, and latency are still reported. Moving caveats to dedicated sections changes emphasis without changing evidence.

**Score:** 7/10, accept.  
**Main concern:** The paper should retain the phrase `among the evaluated systems and metrics` whenever discussing non-dominance.

## Devil's-advocate reviewer

**Assessment:** A reader optimizing Joint F1 and latency can still prefer CrossEncoder. The polished text cannot turn this into a Full win, but it now makes a defensible case that the scientific result is the operating-point and availability-realization analysis.

**Score:** 5/10, weak reject.  
**Main concern:** The contribution may be judged more evaluative than algorithmic.

## Cost/significance reviewer

**Assessment:** The new prose no longer repeats `small` and `failure` language, but the table still exposes the true scale and 1.52x cost. This is an appropriate balance. No efficiency claim is introduced.

**Score:** 5/10, borderline.  
**Main concern:** Full's cost remains hard to justify for applications prioritizing SP/Joint.

## Editorial synthesis

**Decision:** Weak accept / borderline, 6/10. The revision improves clarity and perceived significance by foregrounding replicated positive evidence, strict experimental validity, and the multi-objective contribution. It does not cross into outcome concealment. The strongest remaining rejection risk is novelty: the paper's analysis contribution is more compelling than Full's incremental algorithmic advantage.

## Mandatory final checks

1. Keep all frozen values byte-consistent with the v8 baseline paper.
2. Preserve CrossEncoder's higher SP/Joint and lower latency in Abstract, Main Results, and Conclusion.
3. Keep Answer-drop and Full latency visible in tables and Limitations.
4. Do not add `significant` unless directly supported by the paired intervals/tests.
5. Keep 2Wiki as a boundary, not a success claim.
