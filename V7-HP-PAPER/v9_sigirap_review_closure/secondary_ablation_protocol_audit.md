# Secondary Ablation Protocol Audit

## Decision

**Deferred, with the missing comparison disclosed.** The inventory found no pre-holdout frozen Full-without-pair, Full-without-chain, or Full-without-CrossEncoder checkpoint/action set compatible with the frozen selector and coverage contract.

## Why a new run would be invalid as confirmation

The holdout outcomes and the selected Full architecture are already known. Training a removal variant now, choosing its feature schema, or repairing its operating point after observing either holdout would make the comparison post-inspection and architecture-adaptive. We therefore do not run or report such a model as a confirmatory ablation.

## Evidence retained

The nested-development generator audit was prepared without holdout use and measures action-level opportunity density, query coverage, and answer-safe action rate. It remains a mechanism diagnostic. The separately frozen Lite architecture comparison is also excluded because it changes several modules together and is not a one-component Full removal.

## Prohibited interpretations

- The diagnostic does not show that pair complementarity, chain actions, or CrossEncoder features are necessary.
- Missing cells are not zero effects.
- No post-hoc removal is described as pre-specified or independently confirmatory.
- The primary Full operating point and both holdouts remain unchanged.
