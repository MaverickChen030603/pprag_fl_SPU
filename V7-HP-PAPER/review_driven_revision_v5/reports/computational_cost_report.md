# Computational Cost Report

## Deployment Answer

At inference time the system executes the answer reader **once per query**, on the final selected context. Reader outcomes for candidate actions are generated only during offline supervised development and are not deployment-time reader calls.

## Exact Paired Effects on the Frozen 3,000 Queries

| Population | N | Coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Answer gain / 100 | Joint gain / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All queries | 3000 | 1.0000 | +0.0088 | +0.0056 | +0.0064 | +0.88 | +0.64 |
| Selected interventions | 774 | 0.2580 | +0.0340 | +0.0219 | +0.0250 | +3.40 | +2.50 |
| Fallback | 2226 | 0.7420 | +0.0000 | +0.0000 | +0.0000 | +0.00 | +0.00 |

## Final Reader Runtime (Generator Excluded)

The values below are measured after a context has been constructed. End-to-end online latency remains `[NEEDS MEASUREMENT]` because no comparable generator/encoder timing harness is available for every system.

| System | Reader mean | Reader P50 | Reader P95 | Generator | End-to-end total | GPU memory | Reader calls | Cross-encoder calls | Context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_top5_baseline | 0.1554 | 0.1242 | 0.2719 | [NEEDS MEASUREMENT] | [NEEDS MEASUREMENT] | 2101469696 | 1 | 0 | 669.0 |
| full_v4 | 0.1419 | 0.1225 | 0.2148 | [NEEDS MEASUREMENT] | [NEEDS MEASUREMENT] | 2107716096 | 1 | 10 | 652.1 |
| lite_method | 0.1377 | 0.1214 | 0.2067 | [NEEDS MEASUREMENT] | [NEEDS MEASUREMENT] | 2107716096 | 1 | 0 | 665.7 |
| recomp_top1 | 0.1232 | 0.1176 | 0.1924 | [NEEDS MEASUREMENT] | [NEEDS MEASUREMENT] | 1934474240 | 1 | 0 | 47.6 |
| recomp_budgetmatched | 0.1272 | 0.1176 | 0.2096 | [NEEDS MEASUREMENT] | [NEEDS MEASUREMENT] | 2037304320 | 1 | 0 | 639.2 |

## Offline Development

- Full V4 action reader evaluations: `8934`
- Lite new unique reader evaluations: `1850`
- Full V4 action-label storage: `51603634` bytes
- Total historical GPU hours: `[NOT AVAILABLE]`
- Generator/selector/support training wall times: `[NEEDS MEASUREMENT]` unless recovered from an explicit timing manifest.

Offline action labeling is expensive but amortized; it must not be described as online latency. The deployment scope is bounded post-retrieval pools and moderate-latency or offline QA, not streaming web-scale RAG.
