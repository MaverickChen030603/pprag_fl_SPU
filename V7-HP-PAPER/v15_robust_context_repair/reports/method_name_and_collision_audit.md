# V15 Method Name and Collision Audit

Date: 2026-07-21. Status: initial audit complete; final name not yet frozen.

## Exact-name result

No directly matching research method was found for the exact provisional phrase
`Robust Risk-Calibrated Context Repair` or `Reader-Robust Risk-Calibrated
Context Repair` in the initial exact-phrase searches.

## Nearby concepts that constrain naming and claims

- Context-Picker already frames context selection as minimal sufficient subset
  selection with multi-stage reinforcement learning and leave-one-out labels
  ([arXiv:2512.14465](https://arxiv.org/abs/2512.14465)). V15 must distinguish
  frozen-pool complete-sequence repair, multi-reader direct deltas, exact
  fallback, and independently calibrated per-query intervention.
- Influence Guided Context Selection directly studies learned context
  contribution, so V15 cannot claim to originate outcome-aware context scoring
  ([OpenReview paper](https://openreview.net/pdf/142a1ef32ce0fda6a4a145b78621d8280d743f42.pdf)).
- `risk-calibrated` is used in other areas and in recent retrieval/query-routing
  work. The phrase alone is not a sufficiently distinctive contribution.
- The acronym `R3C` is heavily overloaded outside RAG and is not recommended,
  even though no exact RAG-method collision was located.

## Current recommendation

Retain `Robust Risk-Calibrated Context Repair` only as an internal working name.
Do not freeze a paper acronym before the direct-delta scorer, calibrated gate,
and multi-reader result establish which mechanism actually survives. Candidate
final naming should foreground the empirically validated differentiator rather
than the broad words `robust`, `risk`, or `repair` alone.

## Claim boundary

The literature audit supports novelty investigation, not a novelty conclusion.
A final collision audit must include ACL Anthology, ACM DL, DBLP, arXiv,
OpenReview, and official code repositories for SetR, Contextual Passage Utility,
Influence Guided Context Selection, Context-Picker, R-CPS, RankRAG, and nearby
selective/conformal RAG work before submission.

