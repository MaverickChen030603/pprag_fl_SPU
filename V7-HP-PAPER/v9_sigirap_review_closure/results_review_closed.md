## 5. Main Results

### 5.1 Frozen Same-Source Holdouts

| Frozen split | N | Coverage | Answer delta | SP delta | Joint delta | Joint 95% CI | Paired p | Answer-drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original holdout | 3,000 | 25.8% | +0.0088 | +0.0056 | +0.0064 | [+0.0027,+0.0104] | 0.0004 | 7.75% |
| Revision holdout | 3,405 | 25.9% | +0.0116 | +0.0061 | +0.0080 | [+0.0044,+0.0116] | <0.0004 | 7.83% |

Full improves Answer, SP, and Joint F1 on both frozen holdouts. On the original holdout, baseline/Full values are 0.6183/0.6271 for Answer, 0.4930/0.4987 for SP, and 0.3292/0.3356 for Joint. On the revision holdout they are 0.6129/0.6244, 0.4862/0.4923, and 0.3201/0.3280. Paired 95% intervals exclude zero for all six population deltas. Because both samples come from HotpotQA distractor validation, they provide same-source replication rather than external generalization.

Across the two disjoint frozen same-source evaluations, Answer, SP, and Joint all move in the same positive direction, and all six paired intervals exclude zero; Table 1 exposes the Joint interval and p-value directly. Full modifies about 26% of contexts. Most selected queries tie the baseline, while the Answer- and Joint-drop rates report observed intervention risk at this frozen operating point.

### 5.2 Answer-Evidence Trade-off against a Shared-Checkpoint Reranker

The protocol-matched shared-checkpoint CrossEncoder-Top5 baseline scores every document in the same approximately ten-document pool and selects and orders five using only relevance. Full uses the same frozen relevance checkpoint as one feature among lexical, entity, opportunity, complementarity, and structural signals; that feature does not itself choose Full's context. The baseline excludes pair, missing-hop, opportunity, preservation, utility, and action-family logic. Score order is chosen on development only and then frozen. Because this comparison was added after the primary study, it is a post-hoc secondary baseline analysis. The comparison isolates the value of the complete context-construction and selective-intervention pipeline beyond using the same relevance checkpoint as an independent document ranker. It is protocol-matched, not representation-level independent.

| Split | System | Answer F1 | SP F1 | Joint F1 | Latency (ms/query) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original 3,000 | Frozen Top-5 | 0.6183 | 0.4930 | 0.3292 | 140.88 |
| Original 3,000 | CrossEncoder-Top5 | 0.6078 | 0.5240 | 0.3420 | 149.90 |
| Original 3,000 | Full | 0.6271 | 0.4987 | 0.3356 | 213.48 |
| Revision 3,405 | Frozen Top-5 | 0.6129 | 0.4862 | 0.3201 | 140.88 |
| Revision 3,405 | CrossEncoder-Top5 | 0.6063 | 0.5220 | 0.3405 | 149.90 |
| Revision 3,405 | Full | 0.6244 | 0.4923 | 0.3280 | 213.48 |

The matched comparison exposes a genuine multi-objective result. CrossEncoder moves further on SP and Joint, while changing Answer F1 relative to baseline by -0.0105 and -0.0066. Full improves both Answer and Joint over baseline and reaches Answer F1 +0.0193/+0.0181 above CrossEncoder, at higher latency. CrossEncoder minus Full Joint F1 is +0.0064 on the original holdout (95% CI [-0.0033,+0.0156], p=0.1884) and +0.0124 on the revision holdout ([+0.0034,+0.0211], p=0.0068). Among the evaluated systems and metrics, Full is a non-dominated Answer-oriented operating point and CrossEncoder is a non-dominated evidence-oriented point; neither dominates Answer, Joint, and latency simultaneously.

![Answer-Joint-latency operating points](answer_joint_latency_tradeoff.pdf)

**Figure 2:** Frozen operating points. RECOMP-660 appears only on the original holdout and Lite only on the revision holdout because no corresponding frozen result exists on the other split. Non-dominance is assessed only among the evaluated systems, metrics, and measured post-retrieval latency.

The paired outcomes make the aggregate trade-off more precise. CE improves SP while lowering Answer on 63 (2.1%) and 74 (2.2%) queries. Direct opposition where Full improves Answer and CE lowers it occurs on only 1/0 queries, so the population difference should not be reduced to one common per-query failure mode.

| Post-hoc event | Original 3,000 | Revision 3,405 |
| --- | ---: | ---: |
| CE SP up, Answer down | 63 (2.1%) | 74 (2.2%) |
| Full Answer up, CE Answer down | 1 (0.0%) | 0 (0.0%) |
| Both Joint up | 102 (3.4%) | 127 (3.7%) |

This disagreement analysis uses frozen per-query outcomes and is descriptive. The artifacts do not contain a reliable explicit answer-anchor label, so we do not create an outcome-derived proxy or claim a causal anchor mechanism.

### 5.3 Candidate Opportunity and Selector Regret

The frozen action-set decomposition separates queries with no training-positive action from queries where such an action exists but the policy misses it. A training-positive action is one labeled answer-compatible and utility-improving in the original offline training protocol.

| Split | No training-positive action | Training-positive action missed | Training-positive action selected |
| --- | ---: | ---: | ---: |
| Development 1,000 | 708 | 213 | 79 |
| Original 3,000 | 2,316 | 465 | 219 |
| Revision 3,405 | 2,638 | 515 | 252 |

The same qualitative availability-versus-regret split appears on fully nested development outputs. The decomposition identifies two concrete sources of improvement headroom. Some queries contain no training-positive action in the bounded set; others contain one that the policy does not realize. The retrospective answer-preserving oracle quantifies the remaining action-set headroom but inspects target-query outcomes, so it is a diagnostic rather than a deployable system or fair inference-time competitor. Availability and selection are therefore distinct optimization targets. Full oracle definitions, absolute metrics, gain ratios, regret quantiles, and query-level details remain in the supplement.


A post-hoc fixed-grid support-threshold analysis at 0.5/0.6/0.7/0.8 keeps both Full-baseline SP and Joint deltas positive on both evaluations; the CrossEncoder directions are likewise stable. The pre-specified 0.7 threshold remains unchanged, and the complete table is in the supplement.

## 6. Mechanism and Cost

### 6.1 Core Components

| Generator variant | Positive-action density | Opportunity coverage | Training-label preservation rate | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Full | 14.71% | 29.2% | 92.66% | Frozen joint recipe |
| Without pair complementarity | 10.27% | 27.7% | 93.07% | Clearest learned opportunity loss |
| Without two-document chains | 10.40% | 25.1% | 93.69% | Clearest structural coverage loss |
| Without anchor-preserving families | 16.57% | 27.4% | 92.45% | Higher density but narrower coverage |
| Lite-Lexical-Pair | -- | -- | -- | 0.3217 Joint vs Full 0.3280; NI failed |

Removing pair complementarity or two-document chains produces the clearest development opportunity losses. Removing anchor-preserving families changes both the action denominator and coverage, so its higher positive density is not a monotonic improvement. These outcomes are development opportunity diagnostics consistent with the frozen joint recipe; they do not establish end-to-end component necessity. A clean frozen holdout removal is unavailable because corresponding models were not frozen before holdout inspection, so we do not retrain them post hoc. Opportunity and preservation rates use offline development outcomes for mechanism analysis; they are not inference-time labels or guarantees.

### 6.2 Quality-Risk-Cost Analysis

| System / boundary | Frozen split | Joint contrast | Coverage | Answer-drop | Mean / P95 latency | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Frozen Top-5 | Original 3,000 | reference | 0% | 0% | 140.88 / 252.10 ms | Exact baseline |
| CrossEncoder-Top5 | Original 3,000 | +0.0128 | 100% reranked | -- | 149.90 / 262.59 ms | Higher SP/Joint, lower Answer |
| Full | Original 3,000 | +0.0064 | 25.8% modified | 7.75% | 213.48 / 330.56 ms | Answer-oriented selective point |
| Lite | Revision 3,405 | -0.0063 vs Full | -- | -- | 143.97 / -- ms | Cheaper; NI failed |
| RECOMP-660 | Original 3,000 | -0.0033 vs baseline | 100% compressed | -- | 169.64 / -- ms | Budget control; p=0.4172 |

Full runs its generator and selector for every query even though it modifies only approximately 26% of contexts. It is selective in context modification, not in whether computation is executed. Full adds 72.60 ms/query over baseline, a 1.52x ratio, and is 63.58 ms slower than CrossEncoder-Top5. All evaluated online systems make one final answer-reader call; candidate reader outcomes are offline supervision. Full's mean component times are 70.05 ms for generator, 0.61 ms for selector, and 142.59 ms for serialization plus reader; semantic feature computation dominates the added generator cost.

Lite nearly restores baseline latency but fails the pre-frozen 0.002 Joint-F1 non-inferiority test on the revision holdout. RECOMP-660 uses the same Top-5 input, reader, support predictor, and approximately matched context budget, but its structural action space differs and its Joint change is non-significant. Pair-score pruning provides little latency reduction because semantic feature computation, rather than the number of retained pair actions, dominates generator cost: reducing k from 10 to 3 changes the component-scaled estimate only from 213.48 to 212.04 ms/query. The complete pruning sensitivity remains in the supplement and no pruned method is promoted.

Latency uses one GPU, batch size one, 50 warmup queries, and 500 measured queries with CUDA synchronization. Model loading and upstream retrieval are excluded. These measurements characterize one post-retrieval setup, not throughput, energy, mobile hardware, or a production service-level guarantee.
