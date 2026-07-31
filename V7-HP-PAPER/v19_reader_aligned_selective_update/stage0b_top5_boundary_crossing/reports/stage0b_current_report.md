# V19 Stage 0B Current Report

## Scope

Stage 0B tests whether the BGE LoRA adapter route can convert deeper ranking movement into Top-5 evidence/context improvement before any reader evaluation or selective block upload is allowed.

No calibration or final-test labels are used. Reader evaluation remains disabled.

## PC-1 Boundary Audit

PC-1 hard-negative training completed without hard errors, with loss decreasing from 0.6732 to 0.6596 and unchanged aggregate retrieval:

- support_recall@5: 0.710
- complete_support@5: 0.440
- adapter payload: 1,081,344 bytes

Stage 0B boundary audit found:

- Top-10/20 changed queries: 24/100
- support-promoting among changed queries: 0/24
- support already at rank 6-10: 12/100
- BoundaryOpportunity@5 at threshold 0.02: 4/100
- BoundaryConversionRate: 0/12
- useful Top-5 changes: 0
- harmful Top-5 changes: 0

Interpretation: PC-1 changes rankings, but the movement is mostly irrelevant reorder or deep-support movement. It does not produce reader-visible Top-5 context deltas.

## PC-1 Training Contract Audit

The PC-1 hard-negative manifest contains 1,000 train-only queries and 4,000 explicit negatives:

- entity-overlap negatives: 3,998
- lexical-overlap negatives: 2
- rank 4-10 boundary negatives: 0/4,000
- gold-support false-negative risks: 5
- duplicate negative titles within queries: 6
- possible partial-hop/context negatives: 3,935
- multi-hop queries using only one explicit positive: 992/1,000

Interpretation: PC-1 optimized broad entity-overlap separation, not the rank-5 boundary. It also under-supervised the second support hop.

## PC-2A Boundary Negative Curriculum

PC-2A changed only the negative curriculum toward baseline rank 4-10 boundary negatives while keeping the PC-1 model, LoRA rank, learning rate, steps, payload, and evaluation queries unchanged.

N=100 smoke result:

- loss: 1.1200 to 0.7117
- support_recall@5: 0.715, delta +0.005
- complete_support@5: 0.450, delta +0.010
- Top-5 changed rate: 3%
- Top-10 changed rate: 14%
- Top-20 changed rate: 19%
- support-rank improved/worsened queries: 5/0
- useful/harmful Top-5 changes: 1/0
- complete-support gain/loss: 1/0
- Stage 0B N=100 gate: FAIL, because Top-5 changed rate is below 5%

Interpretation: Boundary negatives move in the right direction but are not strong enough by themselves.

## PC-2B Multi-Positive Support-Aware Loss

PC-2B was run because PC-2A showed positive direction but insufficient Top-5 crossing. PC-2B keeps the same training strength and payload, and changes the objective to multi-positive support-aware loss plus a boundary pairwise term.

N=100 smoke result:

- loss: 0.4825 to 0.3192
- support_recall@5: 0.715, delta +0.005
- complete_support@5: 0.450, delta +0.010
- Top-5 changed rate: 6%
- Top-10 changed rate: 21%
- Top-20 changed rate: 40%
- support-rank improved/worsened queries: 7/0
- useful/harmful Top-5 changes: 1/0
- complete-support gain/loss: 1/0
- boundary conversions: 1
- Stage 0B N=100 gate: PASS

Interpretation: PC-2B is the first minimal variant that crosses the engineering gate. The signal is still small, so it must be confirmed on a disjoint N=300 development subset before any reader evaluation.

## N=300 Retrieval Confirmation

A disjoint development subset has been constructed from HotpotQA development rows 101-400. The confirmation pipeline is running on the server:

- pipeline: `stage0b_top5_boundary_crossing/run_stage0b_n300_confirmation.sh`
- log: `V7-HP-PAPER/v19_reader_aligned_selective_update/logs/stage0b_n300_confirmation.nohup.log`
- pool output: `stage0b_top5_boundary_crossing/retrieval_confirmation/pools/hotpotqa_topic_silo_n300.jsonl`
- reader: disabled
- active decision: wait for N=300 retrieval gate

The pipeline will generate a same-source topic-silo pool, rescore it with Frozen and PC-2B adapters, run boundary audit, and write:

- `retrieval_confirmation/frozen_n300/rescore_summary.json`
- `retrieval_confirmation/pc2b_n300/rescore_summary.json`
- `retrieval_confirmation/pc2b_n300/boundary_audit/pc1_boundary_opportunity_report.md`
- `retrieval_confirmation/n300_gate_summary.json`
- `retrieval_confirmation/retrieval_go_no_go.md`

## Current Decision

Current status: `multi_positive_objective_required`, pending N=300 confirmation.

Reader evaluation must not start yet. If the N=300 gate reproduces positive Top-5 and complete-support movement, the next step is frozen reader-gate diagnostics. If it does not reproduce, the status becomes `development_smoke_overfit`.
