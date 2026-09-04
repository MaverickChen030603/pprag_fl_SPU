# V15 HotpotQA 100-Query Pilot Opportunity

## flan

- Baseline Answer/SP/Joint F1: 0.6553 / 0.4457 / 0.3309
- Joint-oracle Answer/SP/Joint delta: +0.0775 / +0.0228 / +0.0528
- Queries with positive Joint opportunity: 19.00%
- Non-baseline actions causing Joint drop: 14.53%

## unifiedqa

- Baseline Answer/SP/Joint F1: 0.4955 / 0.4457 / 0.2627
- Joint-oracle Answer/SP/Joint delta: +0.0753 / +0.0110 / +0.0474
- Queries with positive Joint opportunity: 15.00%
- Non-baseline actions causing Joint drop: 12.00%

## Same-Action Robust Oracle

| Beta | Intervention | Mean-reader Joint delta | Min-reader Joint delta | Both positive | Any harm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2300 | +0.0481 | +0.0156 | 0.1100 | 0.0100 |
| 0.25 | 0.2200 | +0.0478 | +0.0178 | 0.1100 | 0.0000 |
| 0.5 | 0.2200 | +0.0475 | +0.0185 | 0.1100 | 0.0000 |
| 1 | 0.1100 | +0.0229 | +0.0185 | 0.1100 | 0.0000 |

> All oracle values are retrospective action-set upper bounds and are not deployable selector results.
