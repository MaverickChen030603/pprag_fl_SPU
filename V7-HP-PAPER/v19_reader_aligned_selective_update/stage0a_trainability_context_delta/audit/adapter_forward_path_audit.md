# Adapter Forward-path Audit

Probe texts: 40; cache: bypassed.

| Condition | enabled L2 vs base | disabled L2 vs base | hooks |
|---|---:|---:|---:|
| centralized | 0.10884200 | 0.00000000 | 52 |
| fedavg | 0.09036606 | 0.00000000 | 52 |
| fedprox | 0.08669250 | 0.00000000 | 52 |
| frozen | 0.00000000 | 0.00000000 | 52 |
| scaffold | 0.01642331 | 0.00000000 | 52 |

Forward delta detected: `{'centralized': True, 'fedavg': True, 'fedprox': True, 'frozen': False, 'scaffold': True}`. Disabled outputs should match frozen within numerical tolerance; reload outputs are hashed separately. The LoRA analytic merge is equivalent to the module residual by construction; no persistent merge operation is used in V19.
