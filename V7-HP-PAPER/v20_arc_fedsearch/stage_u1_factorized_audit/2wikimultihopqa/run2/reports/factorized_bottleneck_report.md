# V20 U1 Factorized Audit

Dataset: `2wikimultihopqa`. Final status: `routing_primary_bottleneck_confirmed`. Reader is `blocked_before_reader`.

| Component | Gain | 95% bootstrap CI | Fixed comparator |
|---|---:|---:|---|
| RoutingGain | +0.203 | [+0.160, +0.250] | R1L0 |
| LocalGain | -0.013 | [-0.030, +0.000] | R0L2_hybrid |
| JointGain | +0.187 | [+0.140, +0.237] | R1L2_hybrid |
| Interaction | -0.003 | [-0.010, +0.000] | derived |

Merge residual reactivated: `False`. Oracle is audit-only and uses at most three clients.
