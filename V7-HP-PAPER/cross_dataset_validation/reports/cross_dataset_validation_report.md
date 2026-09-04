# Cross-Dataset Validation Report

## 1. Executive Summary

HotpotQA `selector_v2.3` remains frozen as the paper main result. Cross-dataset validation was initiated, but the first required step, dataset feasibility audit, found no local usable 2WikiMultiHopQA/MuSiQue/IIRC/MultiHop-RAG data with answer/evidence/context fields. Therefore 2Wiki smoke/final and MuSiQue smoke were not run, to avoid fabricating external validation results.

## 2. Dataset Selection Rationale

2WikiMultiHopQA remains the highest-priority external validation target because it is Wikipedia-based and multi-hop. However, it must first be prepared locally with answer, context documents, and evidence/support labels.

## 3. 2Wiki Adapter and Smoke Test

Adapter and smoke-test placeholder outputs were created with status `not_run_dataset_unavailable`. No reader inference was run.

## 4. 2Wiki Final 1000 Result

Not executed. Decision rule requires smoke test success or clear positive trend before running final_1000.

## 5. MuSiQue Smoke Test

Not executed. MuSiQue is second priority and should start only after 2Wiki feasibility is resolved.

## 6. Cross-Dataset Comparison

| dataset | n | method | answer_f1_delta | joint_or_evidence_delta | support/evidence_recall_delta | fallback_rate | positive_candidate_recall | gate_pass | paper_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | 1000 | selector_v2.3 | 0.0023 | 0.015 | 0.019 | 0.5 | 0.3288 | True | main_result |
| 2WikiMultiHopQA | 0 | not_run |  |  |  |  |  | False | pending_external_validation |
| MuSiQue | 0 | not_run |  |  |  |  |  | False | pending_stress_test |


## 7. Failure Analysis

The blocking failure is dataset availability, not method failure. Current evidence does not support any cross-dataset generalization claim.

## 8. Paper Recommendation

Keep the paper main claim HotpotQA-centered. Phrase cross-dataset validation as planned robustness work unless 2Wiki data is prepared and smoke/final results are run.
