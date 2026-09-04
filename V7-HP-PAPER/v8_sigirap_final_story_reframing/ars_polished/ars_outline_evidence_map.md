# ARS Outline and Evidence Map

## Structure pattern

Concise empirical conference paper with a method-plus-analysis contribution.

| Section | Target words | Reader job | Headline evidence |
| --- | ---: | --- | --- |
| Abstract | 180-200 | Establish problem, method, replicated result, trade-off | Two frozen holdouts; CE/Full operating points |
| 1 Introduction | 750-850 | Build availability-realization framing | Frozen deltas, matched CE, oracle decomposition |
| 2 Related Work | 450-550 | Locate gap across retrieval, reranking, construction, selection | Existing citation keys only |
| 3 Method | 850-950 | Explain bounded actions and risk-controlled policy | Inference contract; exact fallback |
| 4 Protocol | 500-600 | Establish validity and fairness | Nested folds, fixed budgets, paired bootstrap |
| 5 Main Results | 1,000-1,150 | Present replicated effects and multi-objective comparison | Absolute F1, CIs, disagreement, decomposition |
| 6 Mechanism and Cost | 500-600 | Explain where opportunity and cost arise | Pair/chain ablations; latency components |
| 7 External Boundary | 150-220 | Bound transfer without interrupting the main claim | 2Wiki aggregate/FDR/calibration |
| 8 Limitations | 300-400 | Consolidate risk, scope, cost | Answer-drop, bounded pool, hardware |
| 9 Conclusion | 160-210 | Reassert contribution and open problems | Replication + operating points + headroom |

## Evidence map

| Claim | Evidence | Placement | Strength label |
| --- | --- | --- | --- |
| Full improves reader outcomes reproducibly | Two disjoint frozen holdouts; all six F1 deltas positive with CIs excluding zero | Abstract, 5.1, Conclusion | Primary confirmatory |
| Full is Answer-oriented among evaluated systems | Highest Answer F1 in matched Baseline/CE/Full table | 5.2 | Secondary post-hoc comparison |
| CE is stronger on SP/Joint and faster | Matched CE scores and 149.90-ms timing | 5.2 | Secondary post-hoc comparison |
| Availability and selection are distinct | No-positive/missed/selected counts plus outcome-aware oracle | 5.3 | Retrospective diagnostic |
| Pair/chains expand opportunity | Development ablations | 6.1 | Mechanism evidence |
| Current transfer is unresolved | Non-significant 2Wiki aggregate, no FDR subgroup, missed calibration target | 7/8 | Boundary evidence |

## Transition logic

- Related Work ends with the missing connection between generated action sets and selective realization.
- Method operationalizes that connection; Protocol establishes why downstream labels do not leak.
- Main Results progress from primary replication to matched trade-off to diagnostic headroom.
- Mechanism and Cost explain how the operating point is produced and what it costs.
- External Boundary and Limitations contain scope qualifications so the main evidence remains readable.
