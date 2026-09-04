# Support-Threshold Sensitivity

This is a post-hoc metric sensitivity analysis over frozen contexts, frozen answer-reader outputs, and frozen support-predictor probabilities. The pre-specified primary threshold remains **0.7**; no threshold is re-selected from these results.

## Original holdout (3,000)

| Threshold | System | SP F1 | Joint F1 | SP delta vs baseline | Joint delta vs baseline |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.5 | Frozen Top-5 | 0.4539 | 0.3068 | +0.0000 | +0.0000 |
| 0.5 | Full | 0.4646 | 0.3158 | +0.0107 | +0.0090 |
| 0.5 | CrossEncoder-Top5 | 0.4822 | 0.3208 | +0.0283 | +0.0140 |
| 0.6 | Frozen Top-5 | 0.4754 | 0.3199 | +0.0000 | +0.0000 |
| 0.6 | Full | 0.4838 | 0.3275 | +0.0084 | +0.0077 |
| 0.6 | CrossEncoder-Top5 | 0.5040 | 0.3318 | +0.0286 | +0.0120 |
| 0.7 **(primary)** | Frozen Top-5 | 0.4930 | 0.3292 | +0.0000 | +0.0000 |
| 0.7 **(primary)** | Full | 0.4987 | 0.3356 | +0.0056 | +0.0064 |
| 0.7 **(primary)** | CrossEncoder-Top5 | 0.5240 | 0.3420 | +0.0309 | +0.0128 |
| 0.8 | Frozen Top-5 | 0.4942 | 0.3286 | +0.0000 | +0.0000 |
| 0.8 | Full | 0.4994 | 0.3355 | +0.0052 | +0.0069 |
| 0.8 | CrossEncoder-Top5 | 0.5307 | 0.3447 | +0.0365 | +0.0162 |

- **Full:** SP direction is stable; Joint direction is stable across the fixed grid.
- **CrossEncoder-Top5:** SP direction is stable; Joint direction is stable across the fixed grid.

## Revision holdout (3,405)

| Threshold | System | SP F1 | Joint F1 | SP delta vs baseline | Joint delta vs baseline |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.5 | Frozen Top-5 | 0.4556 | 0.3041 | +0.0000 | +0.0000 |
| 0.5 | Full | 0.4644 | 0.3134 | +0.0088 | +0.0093 |
| 0.5 | CrossEncoder-Top5 | 0.4805 | 0.3172 | +0.0249 | +0.0130 |
| 0.6 | Frozen Top-5 | 0.4746 | 0.3140 | +0.0000 | +0.0000 |
| 0.6 | Full | 0.4825 | 0.3233 | +0.0079 | +0.0093 |
| 0.6 | CrossEncoder-Top5 | 0.5030 | 0.3304 | +0.0284 | +0.0164 |
| 0.7 **(primary)** | Frozen Top-5 | 0.4862 | 0.3201 | +0.0000 | +0.0000 |
| 0.7 **(primary)** | Full | 0.4923 | 0.3280 | +0.0061 | +0.0080 |
| 0.7 **(primary)** | CrossEncoder-Top5 | 0.5220 | 0.3405 | +0.0358 | +0.0204 |
| 0.8 | Frozen Top-5 | 0.4850 | 0.3178 | +0.0000 | +0.0000 |
| 0.8 | Full | 0.4908 | 0.3256 | +0.0058 | +0.0078 |
| 0.8 | CrossEncoder-Top5 | 0.5270 | 0.3417 | +0.0420 | +0.0239 |

- **Full:** SP direction is stable; Joint direction is stable across the fixed grid.
- **CrossEncoder-Top5:** SP direction is stable; Joint direction is stable across the fixed grid.

## Interpretation boundary

The grid tests whether the reported operating-point comparison is an artifact of the 0.7 support cutoff. It is not a new optimization step and does not alter the frozen Full selector, its coverage budget, the CrossEncoder ranker, or either answer reader.
