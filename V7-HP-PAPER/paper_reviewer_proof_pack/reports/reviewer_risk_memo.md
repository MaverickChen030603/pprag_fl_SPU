# Reviewer Risk Memo

| risk | severity | current_evidence | paper_response | additional_experiment_needed |
| --- | --- | --- | --- | --- |
| single-dataset main result | medium | 2Wiki is included as external diagnostic, not success claim. | Frame paper as HotpotQA-centered with external diagnostic limitation. | no |
| small answer_f1 gain | medium | answer_f1 delta is +0.0023 and non-significant. | Use answer-preserving wording; emphasize joint/support metrics. | no |
| answer_f1 not significant | medium | bootstrap p=0.3625. | Do not claim significant answer_f1 improvement. | no |
| possible test leakage | high | strict no-leak query-level cross-fitting and claim boundary memo. | Describe cross-fitting and separate oracle diagnostics. | no |
| weak cross-dataset selector generalization | high | 2Wiki selector underperforms BM25; detectability audit explains why. | Report 2Wiki as diagnostic limitation. | no |
| oracle upper bound much higher than formal selector | medium | oracle is marked diagnostic-only. | Use oracle as ceiling/future-work motivation only. | no |
| selector may overfit calibration | medium | fold and threshold sensitivity summaries available. | Show fold/config stability and selected_fraction rationale. | no |
| BM25 baseline strong on 2Wiki | medium | BM25 reader smoke strongly beats raw context. | Use BM25 as the correct external baseline. | no |

All listed risks can be handled by wording, tables, and appendix diagnostics. Any additional reader rerun should be treated as optional future work, not required for the current paper cycle.
