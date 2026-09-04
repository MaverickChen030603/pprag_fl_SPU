# V20 U1 Factorized Audit

Dataset: `musique`. Final status: `routing_primary_bottleneck_confirmed`. Reader is `blocked_before_reader`.

| Component | Gain | 95% bootstrap CI | Fixed comparator |
|---|---:|---:|---|
| RoutingGain | +0.230 | [+0.183, +0.280] | R1L0 |
| LocalGain | +0.013 | [-0.010, +0.037] | R0L2_hybrid |
| JointGain | +0.247 | [+0.193, +0.303] | R1L2_hybrid |
| Interaction | +0.003 | [-0.017, +0.023] | derived |

Merge residual reactivated: `False`. Oracle is audit-only and uses at most three clients.
