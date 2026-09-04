# Appendix X: Cross-Dataset Diagnostic on 2WikiMultiHopQA

## X.1 Dataset Adapter and Reader-Backed Smoke

We first validated the 2Wiki adapter and reader-backed evaluation path on dev-300. BM25/lexical routing improved over raw context order by +0.1263 answer-F1, +0.3529 SP-F1, and +0.2507 joint-F1. This establishes the external evaluation pipeline as usable.

## X.2 Selector Alignment against Strong BM25

When evaluated against the strong BM25 baseline, direct Hotpot v2.3 transfer and 2Wiki cross-fitting did not establish selector-level generalization. The appropriate baseline for this dataset is BM25/lexical routing, not raw context order.

## X.3 BM25-Anchor Repair

The BM25-anchor repair preserved the top BM25 answer anchors and reduced negative transfer. The best no-leak repair, `bm25_anchor_answer_neutral_selector`, nearly matched BM25 but produced only +0.0002 joint-F1 delta, so it was not expanded to 1000 samples.

## X.4 Oracle Opportunity and Action Exposure Gap

Oracle diagnostics identified positive actions beyond BM25 for 73 / 300 queries. However, the strict BM25-anchor action table exposed positive actions for only 33 / 300 queries, leaving an action exposure gap.

## X.5 Feature Detectability and Safety Calibration

Feature-margin analysis concludes: positive actions are weakly distinguishable with current features. Safety calibration is weak on 2Wiki, with answer-safe AUC 0.5567 and paper-positive AUC 0.5451.

## X.6 Failure Analysis

The dominant failure mode is candidate-pool limitation: 227 / 300 queries have no oracle positive action beyond BM25. Among strict action-table positive queries, selector recall is 0.3939; many oracle-positive queries do not expose strict positive actions in the no-leak action table.

## X.7 Claim Boundary

2Wiki is reported as an external sanity check and diagnostic limitation. It is not used as a main selector-level generalization claim, and all oracle diagnostics are upper-bound analyses rather than inference-time methods.
