# Generating Reader-Compatible Context Actions for Multi-Hop Question Answering

## Findings-Version Positioning

This fallback presents the same verified experiments with a narrower contribution claim. It is appropriate if a venue or advisor judges the external result too weak for a main-conference method claim.

## Abstract

Context selection cannot improve a multi-hop question when its candidate action set contains no useful, reader-compatible alternative. A controlled heuristic expansion nearly doubles the number of context actions on HotpotQA yet raises positive-query opportunity only from 20.3% to 23.4%, exposing a candidate-opportunity gap. We address this gap with a fully nested pipeline that first generates bounded context actions using missing-hop estimates, semantic document opportunity, pair complementarity, and answer-anchor-preserving construction, and then applies an action only when a reader-safe selector predicts it to be useful without harming the answer. The generator raises positive-query opportunity to 29.2% and positive-action density from 9.43% to 14.71%, although two of five pre-specified opportunity criteria remain unmet. On a 1,000-query development protocol, official answer F1 improves by 0.0133 (p=0.0176) and supporting-fact F1 by 0.0053 (p=0.0372), while joint F1 shows a positive non-significant trend (+0.0064, p=0.0752). Without further tuning, the frozen pipeline improves answer, supporting-fact, and joint F1 by 0.0088, 0.0056, and 0.0064 on 3,000 disjoint same-source HotpotQA queries; the confirmatory joint result is significant (p=0.0004). Answer and joint directions are consistent for FLAN-T5-Large and UnifiedQA-T5-Large. A frozen 2WikiMultiHopQA transfer yields positive but non-significant answer and joint changes and flat support F1, which bounds rather than establishes cross-dataset generalization. These results show that semantic opportunity creation and risk-controlled selection can convert selective context changes into small, reproducible reader gains.

## Core Finding

A selector's ceiling is determined by whether its candidate action set contains an answer-safe evidence improvement. Expanding hand-written actions from 4,000 to 7,882 changes positive-query coverage only from 20.3% to 23.4%. A fully nested semantic generator raises coverage to 29.2% and density to 14.71%, and risk-controlled selection converts this into significant answer/support gains on development and significant answer/support/joint gains on a frozen 3,000-query same-source holdout.

## Evidence

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.483 | 0.6114 | 0.073 | 0.4920 | 0.052 | 0.3241 |
| V4 semantic generation + reader-safe selection | 0.493 | 0.6247 | 0.074 | 0.4973 | 0.051 | 0.3305 |
| Delta | +0.0100 | +0.0133 | +0.0010 | +0.0053 | -0.0010 | +0.0064 |

Paired bootstrap, 5,000 resamples: answer F1 [+0.0024, +0.0249], p=0.0176; supporting-fact F1 [+0.0003, +0.0106], p=0.0372; joint F1 [-0.0005, +0.0132], p=0.0752. Answer F1 and supporting-fact F1 improve significantly at the unadjusted 0.05 level. Joint F1 is positive but not significant. The selector intervenes on 260/1,000 queries and its selected-action answer-drop rate is 5.0%.

| Reader | N | Intervention coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Joint F1 95% CI | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | 3,000 | 0.258 | +0.0088 | +0.0056 | +0.0064 | [+0.0027, +0.0104] | 0.0004 |
| UnifiedQA-T5-Large* | 3,000 | 0.258 | +0.0110 | +0.0056 | +0.0085 | [+0.0049, +0.0122] | <0.0002 |

The 3,000 queries are disjoint from the 1,000-query development sample but come from the same HotpotQA distractor validation source. Generator models, selector thresholds and coverage, reader prompts and decoding, and the support threshold were frozen. FLAN answer F1 p=0.0096; FLAN supporting-fact F1 p=0.0004. *UnifiedQA reuses the frozen selected contexts and the same sentence-support predictor; it is an answer-reader replication, not an independent support-pipeline replication.

The 2Wiki frozen transfer is directionally positive for answer/joint but not significant; support F1 is flat. RECOMP under a standardized FLAN reader is below both the baseline and V4. These analyses support the mechanism and delimit external calibration, but do not establish broad generalization.

## Scope

The paper concerns reader-compatible context action generation and selective intervention. It does not claim a Federated RAG system, privacy preservation, SOTA performance, complete opportunity success, significant development joint F1, or independent support replication across readers.
