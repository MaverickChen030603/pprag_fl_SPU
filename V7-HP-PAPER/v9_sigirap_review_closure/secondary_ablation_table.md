# Secondary Ablation Evidence

## Frozen end-to-end status

| Variant | Original 3,000 | Revision 3,405 | Evidence status |
| --- | --- | --- | --- |
| Full | A/SP/J = .6271/.4987/.3356 | A/SP/J = .6244/.4923/.3280 | Frozen reference |
| Full without pair complementarity | Not legally available | Not legally available | No pre-inspection frozen checkpoint/action set |
| Full without two-document-chain actions | Not legally available | Not legally available | No pre-inspection frozen checkpoint/action set |
| Full without CrossEncoder document feature | Not legally available | Not legally available | No pre-inspection frozen checkpoint/action set |

A clean frozen end-to-end ablation is unavailable because the corresponding model was not frozen before holdout inspection. We therefore retain development opportunity ablations and identify this as a limitation.

## Nested-development opportunity diagnostics

| Variant | Positive-action density | Positive-query coverage | Answer-safe action rate |
| --- | ---: | ---: | ---: |
| Full | 14.7% | 29.2% | 92.7% |
| Full without pair complementarity | 10.3% | 27.7% | 93.1% |
| Full without two-document-chain actions | 10.4% | 25.1% | 93.7% |
| Full without CrossEncoder document feature | 14.7% | 30.6% | 92.6% |

These development values characterize candidate availability inside the bounded generator. They are not end-to-end holdout scores and do not support causal necessity claims.
