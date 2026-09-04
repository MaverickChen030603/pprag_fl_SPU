# SIGIR-AP Rebuttal Templates

## The population effect is small

We agree that the population gains are modest. They are the primary deployment-level results. The larger selected means describe only the subset changed by the frozen policy and do not replace the population effect. We add an outcome-aware diagnostic restricted to the same frozen action sets. On the 3,000/3,405 holdouts, answer-preserving oracle Joint F1 is 0.4397/0.4251, compared with policy 0.3356/0.3280. The decomposition finds 2316/2638 queries without a training-positive action and 465/515 with an available positive action that the selector misses. This analysis is retrospective and is not a deployable result.

## Full costs 1.52x the baseline

We agree that Full introduces measurable cost: 213.48 versus 140.88 ms/query. The selector itself costs 0.61 ms; most overhead comes from generator features. Our development-only pair-pruning sensitivity reports a quality-cost frontier without changing selector thresholds. Retaining three pair evaluations reproduces development Full quality but reduces the component-scaled total only to 212.04 ms/query, indicating that pair scoring is not the main cost. No pruned configuration is promoted because the existing holdouts have already been observed and no independent non-inferiority test remains. The added CrossEncoder-Top5 baseline measures 149.90 ms/query in the same batch-one protocol.

## The 2Wiki result is non-significant

We retain the non-significant aggregate transfer result. The added structural breakdown uses the official 2Wiki type field: compositional, comparison, bridge_comparison, and inference. No type survives BH-FDR correction; the compositional raw p=0.034 becomes 0.136. We therefore do not claim that the method transfers to a particular reasoning type. The accompanying feature-shift analysis is associative, not causal, and we do not continue calibration search after observing target outcomes.

## A strong reranker baseline is missing

We add `ce_score_order`, an independent cross-encoder Top-5 reranker using the same candidate pool, five-document budget, 3,200-character cap, FLAN reader, support predictor, and relevance model, while excluding pair features and reader-outcome selection. The ordering variant is chosen on development only. On the 3,000/3,405 holdouts it reaches Joint F1 0.3420/0.3405, versus Full 0.3356/0.3280. It therefore recovers or exceeds Full's SP/Joint gain, but Answer F1 is lower than Full by 0.0193/0.0181. We revise the claim accordingly: Full is an answer-preserving selective trade-off point, not universally better than neural reranking. Because this baseline was added after the main frozen study, it is labeled post-hoc secondary rather than confirmatory.

## Scope statement

This work establishes a same-source, bounded-pool quality-risk-cost trade-off under one frozen retrieval and reader configuration. The new analyses clarify action-set opportunity, selector regret, independent relevance ranking, and structural transfer boundaries; they do not establish cross-domain robustness, universal reranking superiority, or a low-cost deployment solution.
