# Appendix

## A. Frozen Protocol

- Hotpot source ordering seed: 44.
- Development: 1,000 queries.
- Original confirmatory holdout: 3,000 disjoint queries.
- Untouched revision holdout: 3,405 remaining disjoint queries.
- Baseline: HybridSoftRetriever, alpha 0.55, uniform weights, Top-5.
- Reader: FLAN-T5-Large; 3,200 context characters; 1,024 tokenizer positions; greedy 32-token output.
- Support predictor threshold: 0.7.
- Paired bootstrap samples: 5,000.
- Lite Joint-F1 margin: 0.002, frozen before revision outcomes.

## B. Full Component and Training Details

Full combines lexical and entity features, MPNet similarities, cross-encoder relevance, a missing-hop estimator, a document-opportunity model, pair complementarity, anchor-preserving single and two-document actions, and two selector heads. Each learned component is fold-specific. Outcome labels are computed offline from the frozen reader; inference sees no answer, support label, candidate outcome, or oracle action score.

The central concepts are pair complementarity, bounded two-document chains, anchor preservation, and selective safety. Other modules are included because the joint Full implementation is empirically stronger than Lite, not because every module has been independently validated as a monotonic contribution.

## C. Selected-Policy Distribution

| Holdout | Selected / N | Metric | Mean | Median | Q25 | Q75 | Wins | Losses | Ties | Drop rate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 774/3000 | Answer | +0.0340 | +0.0000 | +0.0000 | +0.0000 | 89 | 60 | 625 | 7.75% |
| Original | 774/3000 | Joint | +0.0250 | +0.0000 | +0.0000 | +0.0000 | 141 | 115 | 518 | 14.86% |
| Revision | 881/3405 | Answer | +0.0447 | +0.0000 | +0.0000 | +0.0000 | 107 | 69 | 705 | 7.83% |
| Revision | 881/3405 | Joint | +0.0309 | +0.0000 | +0.0000 | +0.0000 | 169 | 125 | 587 | 14.19% |

Gain per 100 original-holdout interventions is +3.40 Answer-F1 points and +2.50 Joint-F1 points. These are descriptive accounting quantities, not policy values for unselected queries.

## D. RECOMP Development Curve and Top-1 Diagnostic

The official compressor is evaluated at 64, 128, 256, 384, 512, and 660 token targets on development. The 660-token protocol is frozen for the holdout. The Top-1 compatibility condition averages approximately 47 tokens and one represented document; it is not used for a broad superiority claim. Baseline-Truncated packs source-order sentences to the same targets.

## E. 2Wiki Calibration Grid

The full K-by-method table is stored with the submission artifacts. Each cell averages five fixed seeds and reports coverage, selected answer-drop, Answer/SP/Joint F1, ECE, and Brier score. The minimum mean answer-drop is 5.10%, above the 4% target.

## F. End-to-End Timing Protocol

The timing harness recomputes every online stage while enforcing frozen final contexts. Full is split into document preprocessing, lexical features, MPNet encoding, cross-encoder scoring, missing-hop prediction, document-opportunity scoring, pair-feature construction, pair-complementarity scoring, action construction, safety, positive utility, serialization, and reader. Full mean/P95 total latency is 213.48/330.56 ms, compared with 140.88/252.10 ms for Frozen Top-5. Online reader calls per query equal one for every system.

## G. Reproducibility Boundary

Frozen method predictions, holdout outcomes, and source artifacts are not overwritten. Historical offline GPU-hour totals were not recorded and are therefore unavailable. The timing result covers post-retrieval execution on one fixed machine and should not be read as a retriever or index benchmark.
