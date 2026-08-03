# V20 ARC-FedSearch: Literature-Grounded Direction and Stage U0 Replay

**Date:** 2026-08-03
**Status:** `local_retrieval_first; depth10_smoke_running; reader_forbidden`
**Scope:** This report resets the V17--V19 upstream strategy.  It contains no
new reader result and does not open a final-test split.

## Executive Decision

V17 established that topic partitions change evidence availability but do not
create a stable cross-client context-composition gain.  V19 then established
that retriever adapters can move embeddings and even rankings without reliably
changing reader-visible complete support.  Continuing to tune composer actions,
reader-credit surrogates, selective adapter upload, or stronger LoRA curricula
would optimize downstream symptoms before the upstream recall contract has been
met.

V20 therefore pivots to **ARC-FedSearch (Adaptive Recall-Calibrated Federated
Search)**: data-local query-time resource selection, client-local retrieval,
document-budget allocation, and result merging for multi-hop evidence recovery.
The title is provisional until the new method passes multi-dataset retrieval
gates.

The immediate evidence changes the implementation order proposed at kickoff:

1. Freeze the inherited `Bc=3` source route as a control.
2. Test deeper client-local retrieval and 15-document allocation first.
3. Add source profiles and a recall-first router only if a residual routing gap
   remains after local depth is audited.
4. Add score calibration only if a deeper transmitted list reveals merge loss.
5. Add second-round bridge expansion only if a residual source-availability
   gap remains at the matched mean communication budget.

This is intentionally not a generic six-module stack.  A module with no
independent retrieval contribution is removed.

## What the Reading Changes

| Literature | Actionable lesson | V20 decision |
|---|---|---|
| FeB4RAG | Federated RAG requires separate resource-selection and result-merging contracts, not one undifferentiated retrieval score. | Report every loss stage and compare routing/merge baselines independently. |
| RAGRoute | A deployable source router must be budget- and latency-matched to query-all and fixed routing. | Dynamic routing is a later budgeted baseline, not a free quality model. |
| ReSLLM / SLAT | LLM resource judgments can act as zero/few-shot teachers or synthetic labels. | Use teacher scores only to analyze or distill a train-only lightweight recall-first router; never as a hidden evaluation oracle. |
| ReDDE and classic federated search | Collection size and local result distributions matter; source selection is not centroid similarity alone. | Add multi-prototype/lexical/size profiles only after the local-depth audit. |
| Language-model merging and MKP-QA | Source probability and passage relevance should be combined, but a source prior must not suppress a strong passage from a low-confidence source. | Test smoothed joint source-passage scoring against raw score, RRF, and calibrated merge. |
| Federated LTR / private ranker studies | Non-IID stress and data locality need explicit contracts; local data alone is not a formal privacy guarantee. | Keep physical local indexes, per-client costs, and no-formal-privacy wording. |

RAGRoute reports that lightweight neural source routing can reduce communication
and latency while preserving quality, so V20 must include matched-cost routing
controls rather than treating client contact as free. [RAGRoute](https://arxiv.org/abs/2502.19280)
ReSLLM supplies the strongest rationale for a teacher/student split: its
zero-shot LLM selector and SLAT protocol make resource supervision feasible
without manually labeling every query-resource pair. [ReSLLM](https://arxiv.org/abs/2401.17645)
FeB4RAG reinforces that source selection and aggregation must be evaluated in a
RAG-aware federated-search contract. [FeB4RAG](https://arxiv.org/abs/2402.11891)
MKP-QA motivates later joint domain-passage scoring rather than a hard
domain-first cascade. [MKP-QA](https://aclanthology.org/2025.coling-industry.33/)

## V17/V19 Evidence Retained as Diagnostic Assets

- **V17:** topic-silo Bc=3 gave complete support in the action pool of 41%,
  20%, and 12% for HotpotQA, 2Wiki, and MuSiQue, respectively; topic routing
  exceeded random routing but did not yield robust reader-level synergy.
- **V19:** adapters are not inert: parameter, forward-path, and ranking audits
  passed.  Yet the frozen changed-context reader diagnostic was tied or worse
  than Frozen on 10 intentionally changed contexts.  PC-2B's N=300 retrieval
  gate did produce +1pp complete support at Top-5, but that is a different
  retriever-adaptation route and is now frozen as a diagnostic, not promoted as
  the main method.

The lesson is not that retrieval is hopeless.  It is that any reader-oriented
claim must follow a measured increase in complete evidence access.

## Stage U0: Frozen Inherited Replay, N=100 per Dataset

The replay uses physical V17 topic-silo local indexes and pools.  The current
router's selected clients and local top-5 are frozen.  Gold support is read
only after the pool exists to label loss stages.  No reader call is made.

| Dataset | Centralized retrieval reference complete@20 | Selected clients cover all support | Selected-client local@5 complete | Raw merge@10 complete | Largest observed loss |
|---|---:|---:|---:|---:|---|
| HotpotQA | 0.66 | 0.76 | 0.44 | 0.44 | local retrieval, 0.32 |
| 2WikiMultiHopQA | 0.42 | 0.48 | 0.22 | 0.22 | local retrieval, 0.26 |
| MuSiQue | 0.33 | 0.39 | 0.13 | 0.12 | local retrieval, 0.26 |

The inherited rank-percentile merge did not rescue a single complete-support
case at Top-10.  Raw merge loss was 0pp on HotpotQA and 2Wiki, and 1pp on
MuSiQue.  Therefore score calibration is not the first V20 intervention.

### Interpretation and Caveat

The selected-client coverage is higher than the centralized retrieval reference
in all three cells.  This does **not** mean federated retrieval beats a true
centralized upper bound.  It means that the existing centralized Top-20
retrieval run is a ranked reference whose support recall is not oracle-perfect,
whereas the selected-client metric asks only whether the requisite clients are
reachable.  V20 must use the phrase *centralized retrieval reference* and keep
an explicit gold-only oracle-local upper bound separate.

The U0 replay has a structural limitation: inherited lists stop at local-k=5.
It can identify the current bottleneck but cannot decide whether deeper local
retrieval, dynamic document allocation, or a different hybrid mix would solve
it.  It is therefore a routing decision aid, not a method result.

## Revised Experimental Architecture

### M0: Fixed Route, Deep Local Candidates

Preserve the V17 Bc=3 selected client set.  Materialize all-client local
top-10 with independently computed client-local BGE/BM25 score distributions.
Evaluate source-depth allocations under a strict total transfer budget of 15:
`3x5`, `3x{2,5,8}` allocation, `2x{7,8}`, and source-diverse round-robin.
The first target is complete support and worst-hop rank, not reader F1.

### M1: Client-Specific Hybrid Retrieval

With BGE frozen, compare BM25, dense, fixed 0.55/0.45 hybrid, and a
no-leak query-adaptive mixture based on lexical specificity, entity density,
OOV rate, and dense-sparse disagreement.  The adaptive policy is selected on
train/calibration data only.  It must show local support gain without increasing
the total returned document budget.

### M2: Budget Allocation Before Router Learning

If M0/M1 recover support from depth 6--10, learn or calibrate a source-aware
document allocator at fixed Bc=3 and total docs=15.  Each selected client gets
a minimum one or two documents; additional depth follows source confidence and
uncertainty.  This directly targets the measured loss without claiming a new
source selector.

### M3: Resource Profiles and Recall-First Routing

Only if M2 leaves a material routing availability gap, introduce
multi-prototype dense centroids, lexical collection sketches, and collection
size.  Evaluate ReDDE, single-centroid, classification router, ReSLLM teacher,
and a distilled multi-label student against fixed and random Bc baselines.
The primary labels are complete-evidence-client recall and gold-client recall,
not top-1 client accuracy.

### M4: Calibration, Joint Merge, and Expansion

Enable score calibration only if an M0/M2 transmission list has measurable
support loss at global Top-10.  Enable joint source-passage scoring after a
calibrated merge baseline.  Enable conditional second-round bridge expansion
only if source availability remains the dominant residual loss.  Each change
must retain mean clients contacted <=3 and mean transmitted documents <=15.

## Active Smoke and Next Gate

`hotpotqa_depth10_n100` completed with the frozen V17 topic router and no
reader.  It uses all physical client shards only to materialize candidate lists;
formal metrics consume the same three inherited selected clients per query.

At local-k=5, selected-client complete support is 0.51; at available depth 10
it is 0.58 (+7pp).  Under A0 equal 5/5/5, 15 transmitted documents retain 0.51
complete support, raw cross-client Top-10 falls to 0.32, and rank percentile
recovers 0.46.  The complete matrix selects A1 confidence-proportional
allocation plus rank percentile merge at 0.49, with two byte-identical replay
runs.  The next stage is its pre-specified disjoint N=300 confirmation.

The next decision is mechanical:

- If depth 6--10 rescues at least 5pp complete support before the 15-document
  transfer cap, proceed to M1/M2 (local hybrid plus allocation).
- If extra depth does not rescue support, investigate client-specific hybrid
  retrieval before dynamic routing.
- If deeper candidates create Top-10 merge loss, activate calibration/MKP-style
  joint merge as an ablation.
- Do not start a reader until a frozen N>=300 retrieval configuration improves
  complete support on at least two datasets at matched cost.

## Current Go/No-Go

`promising_but_needs_scale` for a **local-retrieval and budget-allocation**
method.  It is not yet a claim for a full ARC-FedSearch pipeline, a routing-only
method, or a reader-improving RAG method.
