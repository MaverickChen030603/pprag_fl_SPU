# Pair-Complementary Context Actions for Multi-Hop Question Answering

## 6. Computational Cost and Deployment Scope

At deployment, the answer reader is executed once on the selected final context. The reader is not invoked once per candidate action. Candidate reader outcomes are offline labels used to train and audit the generator and selector. The latency columns below measure the final reader after context construction; comparable end-to-end generator latency remains [NEEDS MEASUREMENT].

| System | Offline reader outcomes | Online reader calls | Cross-encoder calls | Reader mean | Reader P95 | Peak memory | Context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_top5_baseline | 0 | 1 | 0 | 0.1554 | 0.2719 | 2101469696 | 669.0 |
| full_v4 | 8934 | 1 | 10 | 0.1419 | 0.2148 | 2107716096 | 652.1 |
| lite_method | 1850 | 1 | 0 | 0.1377 | 0.2067 | 2107716096 | 665.7 |
| recomp_top1 | [NEEDS MEASUREMENT] | 1 | 0 | 0.1232 | 0.1924 | 1934474240 | 47.6 |
| recomp_budgetmatched | [NEEDS MEASUREMENT] | 1 | 0 | 0.1272 | 0.2096 | 2037304320 | 639.2 |

Historical GPU-hour totals and some training-stage wall times are unavailable unless an explicit timing manifest exists; those cells remain marked rather than reconstructed from file modification times. This distinction matters: expensive offline supervision is an amortized research cost, while online deployment consists of feature computation, bounded pair scoring, selector scoring, and one reader call.

The intended use case is an auditable, bounded post-retrieval pool in offline or moderate-latency QA. The method does not address corpus-scale retrieval, streaming index updates, or real-time web search.
