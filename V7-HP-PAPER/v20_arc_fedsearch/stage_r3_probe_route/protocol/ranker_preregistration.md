# R3 Lightweight Probe Ranker: frozen contract

This stage begins only after the R3 Probe-Dev quality gate and compact payload
audit. It uses no reader, no final-test data, no transformer, no LLM, no
reinforcement learning, and no retriever training.

- Probe-Train: the existing R2 `router_train` 5,000-query split.
- Probe-Holdout: the presealed R2-A.6 Recovery-Holdout 300-query split. It was
  created before R2-A.6 results and R2-A.6 never materialized it.
- Candidate generation: frozen P0, `L=8`.
- Features: `static_score` plus the 18 existing R3 scalar probe features.
- Model: class-balanced logistic regression, `C=1.0`, `liblinear`, max 1,000
  iterations, three fixed random seeds `20260807, 20260808, 20260809`.
- Training labels: a candidate client contains at least one support document.
  All P0 Top-8 non-support candidates are retained as high-static hard
  negatives; no random non-candidate negatives are sampled.
- Selection: independent top-3 by predicted probability. Set-aware selection
  is not evaluated in this first supervised stage.
- The label-free comparator is fixed from R3 Probe-Dev: 2Wiki `P5(alpha=.25)`;
  MuSiQue `P1(dense_top1)`.

Fresh-holdout success requires, on both datasets, +8pp coverage@3, +5pp
local-complete@10, and +5pp merged-complete@10 versus static P0, paired
bootstrap lower confidence bounds above zero for coverage and one evidence
metric, no degradation below -2pp, and all three seeds reported. Reader use
remains forbidden until this gate and later cross-dataset requirements pass.
