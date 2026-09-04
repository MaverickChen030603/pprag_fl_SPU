# Final SIGIR-AP Rebuttal Templates

## Why use Full if CrossEncoder has higher Joint F1?

We agree that CrossEncoder-Top5 obtains higher SP and Joint F1 under the matched protocol. It also lowers Answer F1 below the frozen baseline by 0.0105/0.0066, whereas Full improves Answer and Joint on both holdouts. Full costs more. We therefore do not claim universal superiority: the systems occupy different evaluated answer-evidence-latency operating points, and neither dominates all reported objectives.

## Is Answer F1 more important than Joint F1?

No universal metric hierarchy is assumed. We report Answer, SP, and Joint together because they expose different behavior. Full's training objective is oriented toward preserving answer quality while improving joint utility, but an application may prefer the stronger evidence retrieval of CrossEncoder. The paper characterizes this preference rather than resolving it.

## Does the oracle prove that the selector is poor?

Large selector regret is an explicit finding, but the oracle reads target-query reader outcomes and is unattainable online. It is restricted to the frozen action set and separates two limitations: queries without a training-positive action and queries where such an action exists but the policy misses it. We do not claim selector optimality or deployable oracle performance.

## Is pair complementarity necessary?

Removing pair complementarity or bounded chains reduces development opportunity metrics, supporting their role in the frozen action generator. However, strong independent relevance ranking recovers or exceeds Full's SP/Joint result. Our claim is therefore bounded to expanded action opportunity and an Answer-oriented selective operating point, not unique superiority of pair features.

## Is Full too expensive?

Full costs 213.48 ms/query, 1.52x the 140.88-ms baseline and more than the 149.90-ms CrossEncoder. The generator and selector run for every query even though only about 26% of contexts are modified. Lite nearly restores baseline latency but fails the pre-frozen non-inferiority test. We report this cost boundary rather than describing Full as efficient.

## Does 2Wiki demonstrate no generalization?

The aggregate transfer result is non-significant, no official reasoning-type subgroup survives FDR correction, and few-shot calibration misses the 4% Answer-drop target. We therefore make no cross-domain reliability claim. The available taxonomy also does not explain the uncertainty, so further mechanism-aligned analysis requires a separately frozen protocol.
