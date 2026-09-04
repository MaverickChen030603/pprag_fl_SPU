# Outcome-Aware Oracle Diagnostic

## Status and boundary

This analysis is retrospective. It selects only among contexts that were already present in each frozen bounded action set, but it uses target-query reader outcomes and official metrics. It therefore quantifies mechanism potential and selector regret; it is neither a deployable system nor a confirmatory baseline.

## Main results

### Development (nested 1,000)

- Label: **mechanism diagnostic**.
- Baseline / policy / answer-preserving oracle Joint F1: 0.3241 / 0.3305 / 0.4404.
- Available positive-action coverage: 29.2%; positive-action density: 14.7%.
- Frozen policy coverage: 26.0%; aggregate selector capture ratio: 5.5%.
- Mean / P90 / P95 selector regret: 0.1099 / 0.4000 / 0.5000.
- No opportunity / opportunity missed / positive selected: 708 / 213 / 79.

### Original holdout (3,000)

- Label: **post-hoc outcome-aware diagnostic**.
- Baseline / policy / answer-preserving oracle Joint F1: 0.3292 / 0.3356 / 0.4397.
- Available positive-action coverage: 22.8%; positive-action density: 11.9%.
- Frozen policy coverage: 25.8%; aggregate selector capture ratio: 5.8%.
- Mean / P90 / P95 selector regret: 0.1041 / 0.4000 / 0.5000.
- No opportunity / opportunity missed / positive selected: 2316 / 465 / 219.

### Revision holdout (3,405)

- Label: **post-hoc outcome-aware diagnostic**.
- Baseline / policy / answer-preserving oracle Joint F1: 0.3201 / 0.3280 / 0.4251.
- Available positive-action coverage: 22.5%; positive-action density: 12.9%.
- Frozen policy coverage: 25.9%; aggregate selector capture ratio: 7.6%.
- Mean / P90 / P95 selector regret: 0.0971 / 0.4000 / 0.5000.
- No opportunity / opportunity missed / positive selected: 2638 / 515 / 252.

## Interpretation

The diagnostic separates candidate availability from selection. A large no-opportunity segment points to the bounded generator; a large opportunity-but-missed segment points to selector regret. The answer-preserving oracle includes the baseline and cannot be read as an attainable inference-time score. Holdout oracle values are explicitly post-hoc outcome-aware diagnostics and do not validate significance or generalization.
