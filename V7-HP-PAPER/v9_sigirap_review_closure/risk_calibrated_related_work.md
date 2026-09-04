# Risk-Calibrated and Conformal Related Work

## Verified positioning

| Direction | Formal source | What it controls | Difference from Full |
| --- | --- | --- | --- |
| Selective prediction | Geifman and El-Yaniv, ICML 2019 | Empirical risk-coverage through rejection | Full falls back to a retrieval baseline and uses two outcome-supervised heads |
| Risk-controlling prediction sets | Bates et al., JACM 2021 | Finite-sample expected-loss control for set-valued prediction under calibration assumptions | Full reports average observed intervention risk with no statistical guarantee |
| Learn-then-Test | Angelopoulos et al., 2021 | Finite-sample calibration through hypothesis testing | Full's development grid is ordinary empirical model selection, not LTT |
| TRAQ | Li et al., NAACL 2024 | Conformal answer/retrieval prediction sets | Full emits one context and one answer, not a calibrated prediction set |
| C-RAG | Kang et al., ICML 2024 | Conformal generation-risk bounds for RAG | Full's preservation gate is not certified generation-risk control |

Primary source pages:

- SelectiveNet: https://proceedings.mlr.press/v97/geifman19a.html
- Distribution-Free Risk-Controlling Prediction Sets: https://www.gsb.stanford.edu/faculty-research/publications/distribution-free-risk-controlling-prediction-sets
- Learn-then-Test: https://arxiv.org/abs/2110.01052
- TRAQ: https://aclanthology.org/2024.naacl-long.210/
- C-RAG: https://proceedings.mlr.press/v235/kang24a.html

## Safe manuscript language

Our gate is outcome-supervised and empirically calibrated, but it does not provide conformal, PAC-style, per-query, or group-conditional coverage guarantees. It measures average observed intervention risk at one frozen operating point. Conformal alternatives may provide marginal finite-sample guarantees under explicit exchangeability, score, loss, and calibration assumptions. They do not automatically solve domain shift or certify every query.

## Revision decision

No conformal gate is added during this submission cycle. Introducing one after holdout inspection would require a separate calibration protocol and new validation. Distribution-aware risk control is stated as future work rather than implied by the current terminology.
