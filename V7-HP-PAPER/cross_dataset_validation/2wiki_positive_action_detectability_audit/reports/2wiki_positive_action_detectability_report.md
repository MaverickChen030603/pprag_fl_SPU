# 2Wiki Positive Action Detectability Audit

## 1. Executive Summary

This audit does not run 2Wiki-1000 reader validation and does not modify the frozen HotpotQA v2.3 result. It diagnoses why the current no-leak selector fails to reliably identify actions that outperform a strong BM25 baseline on 2Wiki dev-300.

Main conclusion: positive actions beyond BM25 exist, but they are sparse and weakly captured by the current feature/ranker stack. The best BM25-anchor no-leak selector nearly matches BM25 but does not provide enough reliable margin for full 1000-sample validation.

## 2. Existing 2Wiki Results

- Positive-vs-BM25 queries: 73 / 300 (0.2433)
- Best no-leak selector: `bm25_anchor_answer_neutral_selector`
- Best no-leak joint delta vs BM25: 0.0002
- Selected effective action rate: 0.7000
- Positive-vs-BM25 recall: 0.0433

## 3. Oracle Opportunity beyond BM25

Oracle diagnostics show a non-trivial opportunity: oracle answer delta 0.1615, evidence delta 0.1267, and joint delta 0.1533. This remains diagnostic only and is not an inference-time method.

## 4. Candidate Pool Limitation

The candidate pool is the dominant bottleneck: oracle diagnostics mark 227 / 300 queries as having no positive action beyond BM25. Under the stricter action-level label available in the BM25-anchor action table, only 33 / 300 queries expose a positive action to the current selector features. This supports the interpretation that future work should improve candidate generation beyond BM25 rather than simply tuning selector thresholds.

## 5. Positive Action Feature Margin

Feature detectability summary: positive actions are weakly distinguishable with current features.

Top absolute univariate effects:

| feature | effect_size | auc | rho_joint |
| --- | --- | --- | --- |
| num_added_docs | 1.0594 | 0.6797 | -0.0381 |
| answer_risk_score | 1.0342 | 0.6742 | -0.0406 |
| num_removed_docs | 1.0067 | 0.6759 | -0.0456 |
| bm25_score_delta | -0.3624 | 0.2802 | 0.0403 |
| support_proxy_delta_vs_bm25 | -0.2995 | 0.3377 | 0.0169 |
| evidence_proxy_delta_vs_bm25 | -0.2769 | 0.3421 | 0.0078 |


## 6. Selector Recall Failure

Among strict action-labeled positive-vs-BM25 queries, selector positive recall is 0.3939 (13 / 33). At the broader oracle-query level, 49 / 73 oracle-positive queries do not expose a strict positive action inside the BM25-anchor table used by the no-leak selector. Missed positives therefore reflect both candidate/action mismatch and ranker weakness.

Best-positive predicted rank distribution:

| rank | count |
| --- | --- |
| 1 | 3 |
| 3 | 19 |
| 4 | 2 |
| 5 | 2 |
| 6 | 7 |


## 7. Safety Predictor Weakness

The safety predictor is weak cross-dataset: answer-safe AUC 0.5567, paper-positive AUC 0.5451. Its probabilities should not be used as evidence that answer-neutral calibration transfers reliably to 2Wiki.

## 8. Case Studies

Case-study files were exported under `outputs/case_studies/`, covering selected positives, missed positives, BM25-strong cases, and answer-drop selections.

## 9. Paper Recommendation

Freeze HotpotQA v2.3 as the main result. Use 2Wiki as external diagnostic / limitation and appendix evidence, not as main selector-level generalization success. The paper-safe conclusion is that 2Wiki validates the adapter, reader-backed smoke pipeline, and lexical-routing sanity check, while exposing a cross-dataset selector detectability limitation.
