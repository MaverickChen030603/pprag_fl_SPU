# Experiment Sufficiency Memo

Current experiments are sufficient for a HotpotQA-centered paper with 2Wiki diagnostic limitation.

Current experiments are not sufficient for a strong cross-dataset generalization claim.

No further large-scale reader validation is recommended in the current paper cycle.

Only low-cost stability, sensitivity, baseline fairness, and case-study packaging are recommended; these are provided in this proof pack.

## must_include_in_main_paper

- HotpotQA v2.3 final_1000 main result.
- Bootstrap significance table.
- Fair baseline / ablation comparison.
- Claim that answer_f1 is preserved, not significantly improved.

## should_include_in_appendix

- Fold or hash-bucket stability.
- Threshold sensitivity.
- Case studies.
- 2Wiki diagnostic limitation and detectability audit.

## should_not_claim

- Significant answer_f1 improvement.
- Successful 2Wiki selector-level generalization.
- Oracle as an inference-time method.
- Cross-dataset reliability of the safety predictor.

## future_work_only

- 2Wiki 1000 reader validation.
- MuSiQue expansion.
- Candidate generation beyond BM25.
- Cross-dataset safety calibration.
