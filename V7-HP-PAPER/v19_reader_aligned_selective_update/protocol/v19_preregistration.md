# V19 Pre-registration: Reader-Aligned Selective Update for Federated Retrieval

## Status and scope

This is a new project after V17/V18 `checkpoint_a_fail`. It does not re-open,
modify, or attempt to rescue the generic FedAction composer, its action space,
or its personalized selector. The internal working label is **RASU-FedRAG**;
it is not a final paper title. A literature/name audit must precede submission.

## Question

Under a fixed training communication budget, can block selection based on an
independent local-probe reader utility preserve or improve downstream multi-hop
QA compared with random, magnitude, gradient, local-loss, dropout, and the
historical V7-HP selector?

## Frozen evaluation contract

The V17 train/development/calibration/final-test split identities are inherited
unchanged. Stages 0-3 may read only train, development, and calibration. Final
test inputs and labels remain sealed until the method, payload, and credit
weights are frozen. The answer readers, support evaluator, source router,
prompt, top-5 context budget, and reader call budget are frozen.

## Staged gates

1. **Stage 0, full upload:** frozen, local-only, centralized, FedAvg, FedProx,
   SCAFFOLD, and shared/private adapters. Continue only if a federated full
   upload improves retrieval or complete support on both HotpotQA and MuSiQue,
   has a non-negative reader signal, and is not fully dominated by local-only.
2. **Stage 1, credit validity:** exact inclusion and leave-one-block-out probe
   credit must positively align with held-out block QA contribution and beat
   magnitude, gradient, and loss proxies; harm prediction must be useful.
3. **Stage 2, equal payload:** at 20%/25%, compare all same-byte methods. A
   pass requires a quality-preservation, quality-improvement, or communication
   reduction criterion stated in the task brief.
4. **Stage 3+:** uncertainty, conflict, and private adapters are conditional
   additions, not rescue knobs.

## Fixed initial implementation choices

- BGE-base-en-v1.5; frozen base encoder; rank-8 LoRA residuals.
- Thirteen blocks: separate attention-output and FFN-output groups in layers
  6-11, plus the pooler. No post-development change to block granularity.
- M=20, topic-silo alpha=0.1/0.3/1.0, Bc=3 primary and Bc=2 stress setting.
- Three seeds for development; five for the final primary table.
- Query paired bootstrap (5,000 resamples), 95% CI, two-sided paired test;
  BH correction on secondary contrasts, and seed mean/standard deviation.

## Name-conflict note

FedMosaic already studies parametric adapters and selective, non-conflicting
adapter aggregation. V19 must therefore claim neither the first federated RAG
adapter nor generic selective aggregation. Its only candidate contribution is
reader-aligned **parameter-block** credit under fixed communication, evaluated
through a frozen retrieval-to-reader chain.
