# Paper Update Recommendation

## Recommendation

Do not update the paper claim to cross-dataset generalization yet.

## Rationale

The feasibility audit did not find local usable 2WikiMultiHopQA or MuSiQue data. Per the experiment rules, smoke/final evaluations should not be run without adapter-ready answer/evidence/context fields.

## Suggested Paper Wording

Current wording should remain:

> We validate the answer-neutral positive-action selector on HotpotQA and leave cross-dataset validation to future robustness experiments.

If 2Wiki is later prepared and succeeds, update to:

> The answer-neutral positive-action selector generalizes beyond HotpotQA to another Wikipedia-based multi-hop QA benchmark.
