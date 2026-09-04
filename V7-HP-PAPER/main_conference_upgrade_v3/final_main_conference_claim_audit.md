# Final Main-Conference Claim Audit

| Claim | Status | Source artifact | Metric / uncertainty | Dataset | Reader | Protocol |
| --- | --- | --- | --- | --- | --- | --- |
| v2 improves title recall | supported | `submission_revision_v2/nested_significance_report.json` | +0.0120, p=0.007 | HotpotQA-derived 1000 | FLAN-T5-large | fully nested |
| v2 improves title F1 | supported | same | +0.0150, p=0.018 | HotpotQA-derived 1000 | FLAN-T5-large | fully nested |
| v2 improves answer F1 | **not supported** | same | +0.0028, p=0.344 | HotpotQA-derived 1000 | FLAN-T5-large | fully nested |
| v2 improves product | **not supported** | same | +0.0079, p=0.1245 | HotpotQA-derived 1000 | FLAN-T5-large | fully nested |
| v3 candidate generation raises opportunity | descriptively supported | `outputs/action_outcomes/v3_action_outcome_summary.json` | 20.3% to 23.4%; no inferential claim | HotpotQA-derived 1000 | FLAN-T5-large | fixed no-leak generator |
| v3 improves official joint | **not evaluated** | stop-rule artifacts | none | none | none | opportunity gate failed |
| reader robust | **not supported** | multi-reader stage skipped | none | none | none | none |
| cross-dataset generalization | **not supported** | `reports/external_dataset_decision.md` | 2Wiki diagnostic 24.33%, gate fail | 2Wiki dev-300 | FLAN-T5-large diagnostic | smoke only |
| risk-controlled/calibrated v3 | implementation specified, result not run | `04_train_nested_selector_v3.py` | none | none | none | stopped before fitting |
| SOTA | **prohibited** | no faithful comparison | none | none | none | none |

No use of “improves multi-hop QA,” “robust across readers,” “generalizes,” “official joint gain,” or “SOTA” is permitted for v3.
