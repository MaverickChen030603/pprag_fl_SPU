# V15 Cross-Reader Robust Utility Diagnostic

> Reader labels are used only for retrospective evaluation and the explicitly labelled oracle. Learned selections use inference-safe features only.

## Reader Agreement

- Joint-delta Spearman: 0.4888
- Joint-delta Pearson: 0.2855
- Non-zero sign disagreement: 0.0417

## Robust Selection

| Selector | Intervention | Mean-reader Joint delta | Min-reader Joint delta | Both positive | Any harm |
|---|---:|---:|---:|---:|---:|
| learned_beta_0 | 0.5909 | -0.0058 | -0.0365 | 0.0909 | 0.0909 |
| oracle_beta_0 | 0.2727 | +0.0349 | +0.0132 | 0.1364 | 0.0455 |
| learned_beta_0.25 | 0.5909 | -0.0172 | -0.0547 | 0.0455 | 0.0909 |
| oracle_beta_0.25 | 0.2273 | +0.0334 | +0.0233 | 0.1364 | 0.0000 |
| learned_beta_0.5 | 0.5000 | -0.0172 | -0.0547 | 0.0455 | 0.0909 |
| oracle_beta_0.5 | 0.2273 | +0.0334 | +0.0233 | 0.1364 | 0.0000 |
| learned_beta_1 | 0.4545 | +0.0113 | +0.0022 | 0.0455 | 0.0000 |
| oracle_beta_1 | 0.1364 | +0.0264 | +0.0233 | 0.1364 | 0.0000 |
| exact_fallback | 0.0000 | +0.0000 | +0.0000 | 0.0000 | 0.0000 |

## Interpretation

The oracle rows measure action-set opportunity, not deployable performance. The learned rows are the valid checkpoint test: a positive robust delta must be realized by one label-free action across both readers.
