## 5. Main Results

### 5.1 Two Frozen Same-Source Holdouts

| Split | N | Coverage | Baseline A-F1 | Full A-F1 | A delta (95% CI; p) | SP delta (95% CI; p) | Joint delta (95% CI; p) | Selected A-drop |
|---|---:|---:|---:|---:|---|---|---|---:|
| Original frozen holdout | 3000 | 25.8% | 0.6183 | 0.6271 | +0.0088 ([+0.0023, +0.0152]; 0.0096) | +0.0056 ([+0.0031, +0.0083]; 0.0004) | +0.0064 ([+0.0027, +0.0104]; 0.0004) | 7.75% |
| Untouched revision holdout | 3405 | 25.9% | 0.6129 | 0.6244 | +0.0116 ([+0.0052, +0.0178]; <.0002) | +0.0061 ([+0.0036, +0.0088]; <.0002) | +0.0080 ([+0.0044, +0.0116]; <.0002) | 7.83% |

Full improves all three F1 measures on both holdouts. On the original 3,000-query holdout, the paired deltas are +0.0088 Answer, +0.0056 SP, and +0.0064 Joint F1. The untouched 3,405-query holdout confirms +0.0116, +0.0061, and +0.0080. The latter simultaneously serves as the independent Lite non-inferiority test and a replication of Full. Both sets are disjoint from development, Full was frozen before both runs, and revision outcomes were unread when the Lite architecture was fixed. Because both are HotpotQA same-source samples, we do not pool them for a new significance claim.

### 5.2 Descriptive Effects on Policy-Selected Interventions

Population and conditional views answer different questions. In the original holdout, Full edits 774/3000 contexts (25.8%); Answer/SP/Joint population deltas are +0.0088/+0.0056/+0.0064. Conditional on these policy-selected interventions, the descriptive means are +0.0340/+0.0219/+0.0250. Answer has 89 wins, 60 losses, and 625 ties; Joint has 141/115/518. The selected Answer- and Joint-drop rates are 7.75% and 14.86%. Medians and both interquartile endpoints are zero because most selected contexts tie the baseline.

The revision holdout shows the same concentration pattern: 881/3405 interventions, with descriptive selected Answer/SP/Joint means of +0.0447/+0.0237/+0.0309. These values are descriptive gains conditional on policy-selected interventions. They are not causal treatment effects, expected gains for arbitrary queries, or effects on all improvable queries. In both holdouts, fallback contexts and metrics are exactly unchanged.

### 5.3 Full-to-Lite Non-Inferiority

| Revision-holdout system | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|
| Frozen Top-5 | 0.6129 | 0.4862 | 0.3201 |
| Full | 0.6244 | 0.4923 | 0.3280 |
| Lite-Lexical-Pair | 0.6149 | 0.4860 | 0.3217 |

Lite minus Full Joint F1 is -0.0063 (95% CI [-0.0104, -0.0023], p=0.0004). It misses both the point and interval versions of the frozen 0.002 margin. Lite reduces computation, but the independent quality criterion fails; it is therefore a simplification diagnostic rather than a replacement for Full.

## 6. Budget-Matched Compression

| System (3,000 holdout) | Tokens | Documents | Answer F1 | SP F1 | Joint F1 | E2E ms/query |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Top-5 | 664.5 | 4.986 | 0.6183 | 0.4930 | 0.3292 | 140.88 |
| Baseline-Truncated-660 | 635.7 | 4.236 | 0.6038 | 0.4904 | 0.3224 | 147.43 |
| RECOMP-660 | 635.9 | 4.873 | 0.6226 | 0.4837 | 0.3259 | 169.64 |
| Full | 656.1 | 4.986 | 0.6271 | 0.4987 | 0.3356 | 213.48 |

RECOMP-660 changes Answer/SP/Joint F1 relative to Frozen Top-5 by +0.0043/-0.0093/-0.0033. The Joint interval is [-0.0109, +0.0044], p=0.4172; the difference is not significant. Under an approximately matched context budget and a standardized FLAN reader, RECOMP-style sentence packing does not improve the frozen multi-hop baseline, whereas Full context actions retain a positive same-source effect. This is an official-compressor implementation under reader and budget adaptation, not a claimed end-to-end reproduction. Matched tokens also do not create identical structural action spaces: sentence compression and pair-complementary five-document actions optimize different objectives. The approximately 47-token Top-1 condition is retained only as a compatibility diagnostic in the appendix.

## 9. Analysis

**Opportunity before selection.** The action generator determines whether repair is possible at all. Pair complementarity raises the chance that a proposal contains both hops, while bounded construction prevents opportunity from becoming an uncontrolled permutation search. Selection then trades coverage for answer risk. This explains why conditional gains can exceed population gains without implying a broad treatment effect.

**What the Lite failure means.** Pair complementarity, chains, anchors, and selective safety are the most interpretable mechanisms. Yet the untouched holdout shows that lexical pair features alone do not preserve Full quality within the chosen margin. Missing-hop, MPNet, cross-encoder, and document-opportunity components therefore remain in the stronger implementation. Their mixed individual ablations support neither a claim that each is necessary nor a claim that each always helps.

**Compression versus structured action.** Equalizing token budget removes the most obvious information-volume confound, but it does not equalize objectives. Sentence packing chooses text spans; Full selects a small structural intervention while retaining five-document coverage. The comparison bounds interpretation rather than identifying one universally better constructor.

**Transfer as a gate boundary.** 2Wiki retains some positive candidate opportunity, but the Hotpot safety probabilities are misaligned with target-domain harm. Target-train calibration lowers risk only partially. The current evidence therefore separates reusable action construction from unresolved risk calibration under shift.
