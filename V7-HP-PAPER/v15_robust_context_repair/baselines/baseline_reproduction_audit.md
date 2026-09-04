# V15 Baseline Reproduction Audit

Status: protocol contracts frozen; numerical reproduction pending.

| Method | Same real pool | Five-doc budget | Same readers | No target tuning | Online reader calls | V15 decision |
|---|---|---|---|---|---:|---|
| Frozen Top-5 | Yes | Yes | Yes | Yes | 1 | Required |
| CrossEncoder-Top5 | Yes | Yes | Yes | Yes | 1 | Required |
| V14 Full | To port | Yes | Yes | Must audit | 1 | Required |
| RECOMP | To port | Yes | Yes | Must audit | 1 | Required |
| Source-order truncation | Yes | Yes | Yes | Yes | 1 | Required |
| Marginal utility selector | To implement | Yes | Yes | Yes | 1 | Required |
| Direct-delta MLP | Yes | Yes | Yes | Yes | 1 | Required |
| Fixed-pool subset utility | To implement | Yes | Yes | Yes | 1 | Required |

Adjacent methods are compared numerically only when official code can preserve
the same frozen pool, document budget, readers, retriever, and online-call
contract. Otherwise they remain in the contract matrix and related work.

