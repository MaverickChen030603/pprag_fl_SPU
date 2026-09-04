# Abstract / Introduction / Contributions Draft

## Abstract Draft

Federated routing exposes support-relevant context candidates for multi-hop question answering, but naive context action insertion can hurt reader answer quality. This creates a policy-action-to-reader gap: actions that look useful from a routing or support perspective may not improve downstream answer-bearing reasoning. We propose answer-neutral positive-action selection, a no-leak action selector that applies routed context actions only when they are predicted to preserve answer quality while improving joint/support utility. Under strict no-leak query-level cross-fitting on HotpotQA, the method significantly improves joint_f1 and support-side metrics while preserving answer_f1 with a small non-significant positive delta. A 2WikiMultiHopQA diagnostic further shows that cross-dataset selector transfer remains limited by candidate exposure and safety calibration, motivating future work on dataset-robust action generation.

## Introduction Positioning

This paper studies the downstream action-selection problem after federated routing in multi-hop RAG. Federated retrieval can surface contexts with useful support evidence, yet these contexts are not automatically beneficial to a reader. The central question is therefore not only whether a client can discover useful evidence, but whether an action-selection policy can decide when adding that evidence will preserve answer quality and improve joint reasoning.

## Contributions

1. We identify the policy-action-to-reader gap in federated RAG for multi-hop QA.
2. We propose an answer-neutral positive-action selector under strict no-leak cross-fitting.
3. We provide main HotpotQA results, ablations, and 2Wiki diagnostics showing both effectiveness and cross-dataset limitations.
