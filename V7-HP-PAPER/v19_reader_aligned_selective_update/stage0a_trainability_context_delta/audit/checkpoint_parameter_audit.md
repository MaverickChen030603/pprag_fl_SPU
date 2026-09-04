# Checkpoint Parameter Audit

Reference: `frozen`; checkpoints: centralized, fedavg, fedprox, frozen, scaffold.

| Condition | Non-zero delta vs frozen | Checkpoint hash |
|---|---:|---|
| centralized | True | `644feec8820d` |
| fedavg | True | `5796c767e9d6` |
| fedprox | True | `813212f69a97` |
| frozen | False | `d8963de19f96` |
| scaffold | True | `c51a24993d0a` |

Gradient and exact optimizer-step telemetry were not recorded by the original smoke runner. This is an audit finding: later Stage 0A/positive-control runs must persist them. Existing non-zero parameter deltas are sufficient to proceed to the forward-path audit, not to a scientific effectiveness claim.
