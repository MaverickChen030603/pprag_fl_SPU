# Weekly Research Progress Report

**Week:** 2026.07.27-2026.07.31  
**Project:** Federated RAG / Multi-hop QA  
**Primary work:** V17 Federated Action RAG Checkpoint-A and V18 protocol preparation

## 1. Weekly objective

This week aimed to determine, before training a more complex federated model,
whether natural client partitions actually create enough **budget-realizable,
reader-beneficial cross-client context-composition opportunity**. The question
is deliberately narrower than whether distributed retrieval can expose more
documents: the evidence must remain after a fixed client budget, a fixed K=5
reader context, two answer readers, and matched centralized/random controls.

## 2. Experiment progress

### 2.1 Completed work

- Built and ran the formal V17 Phase-A checkpoint on three multi-hop QA
  datasets: HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- Used two frozen readers: FLAN-T5-Large and UnifiedQA-T5-Large.
- Completed all **30** dataset-reader-condition cells with **N=100** queries
  each. Conditions were centralized, topic-silo at Bc=2, topic-silo at Bc=3,
  entity-community at Bc=3, and random-balanced control at Bc=3.
- Kept the formal contract fixed from commit `fac9f62`: M=20, local k=5 for
  federated conditions, a ten-document action pool, K=5 final contexts, and a
  local sparse candidate cap of 20. Centralized retrieval correctly used its
  fixed k=10 control contract.
- Implemented physical client-local FTS shards, source-routing metadata,
  federated context generation, two-reader Oracle labeling, control analysis,
  and an automatic Go/No-Go aggregator.
- Added V18 preparation materials: a decision-object literature matrix,
  collision audit, Checkpoint-A integrity audit, and unified post-completion
  reporting scripts. The integrity audit passed **104/104** checks, covering
  cell completeness, query alignment, budgets, context size, document
  uniqueness, client IDs, provenance, hashes, and no-leak rules.
- Static no-leak audit passed: no final-test labels, gold support, answer
  presence, or reader-generated target outcomes were available to routing or
  future inference-time features.

### 2.2 Main result

The preregistered V17 decision is **`hold_or_redirect`**; the V18 branch is
**`checkpoint_a_fail`**. This is a useful negative result, not a system error.

| Primary topic-silo Bc=3 result | FLAN StrictSyn (95% CI) | Comp-only | UnifiedQA StrictSyn (95% CI) | Comp-only |
|---|---:|---:|---:|---:|
| HotpotQA | +0.0338 [-0.0098, +0.0753] | 7% | +0.0292 [-0.0078, +0.0648] | 4% |
| 2WikiMultiHopQA | -0.0342 [-0.0895, +0.0197] | 5% | -0.0078 [-0.0582, +0.0425] | 9% |
| MuSiQue | +0.0234 [-0.0248, +0.0703] | 10% | +0.0288 [-0.0133, +0.0716] | 11% |

No dataset satisfied the full requirement on both readers: positive StrictSyn
with a confidence-interval lower bound above zero, composition-only rate of at
least 10%, superiority over both centralized and random controls, and positive
Bc=2 evidence. Therefore Phase B centralized composer learning, Phase C
FedAvg/FedProx/SCAFFOLD/personalization, and Phase D selective model upload are
**not allowed** under the frozen protocol.

### 2.3 Interpretation

- Natural topic silos did create cross-client exposure: the Bc=3 cross-client
  evidence rates were 43% (HotpotQA), 63% (2Wiki), and 74% (MuSiQue).
- However, complete support in the ten-document action pool was only 41%, 20%,
  and 12%, respectively. More importantly, that exposure did not consistently
  become reader-level strict synergy.
- Topic-silo composition-only rates were sometimes higher than random controls
  (for example MuSiQue-FLAN: +10 percentage points, CI [+5, +16]), but did not
  beat matched centralized controls. Hence the observed signal is insufficient
  for the stronger claim that federation creates uniquely learnable reader
  context-composition opportunity.
- Entity-community partitions were mostly negative on StrictSyn. This makes it
  inappropriate to rescue the method by post-hoc partition switching.

## 3. Literature reading and synthesis

I reorganized the literature around the *decision object*, rather than treating
all papers as directly comparable Federated RAG baselines.

| Reading group | Representative work | Key lesson for the project |
|---|---|---|
| Training-time update selection | FedAvg, Federated Dropout, Adaptive Federated Dropout, FedPAQ, prior V7-HP selective upload | Select model parameters/blocks during training; this is distinct from query-time context composition. It remains the most defensible next direction. |
| Federated retriever adaptation | pFedRAG, FedRAG framework, FedE4RAG | Focus on shared/personalized retriever or RAG component parameters, not a complete reader-context action. Personalization needs its own confirmed opportunity signal. |
| Query-time source routing | RAGRoute / Efficient Federated Search, FeB4RAG, DRAG | Source routing and result merging are meaningful components and baselines, but source access alone does not establish downstream reader gains. |
| Aggregation and conflict | FedMosaic and conventional result merging | Selective aggregation and conflict control are useful inspirations, but FedMosaic operates in adapter/parameter space; it cannot be claimed as a direct document-context baseline. |
| Distributed memory / agents | FD-RAG, HyFedRAG, Federated In-Context LLM Agent Learning | Fast/slow paths and distributed memory are relevant system designs, yet their shared artifact and task contracts differ from fixed K=5 reader-context actions. |
| Evaluation | RAGAS and MSRS | RAGAS should remain auxiliary. Dataset-native answer/evidence metrics and controlled reader evaluation must remain primary; MSRS is a later external multi-source synthesis check, not a replacement for multi-hop QA evidence. |

The closest methodological collisions are reader-aware centralized set/context
selection (e.g., SetR, Context-Picker, contextual utility and influence-guided
selection) and federated retrieval/personalization (pFedRAG/FedRAG). The
project may not claim “first selective federated RAG,” “first source routing,”
or “first distributed evidence composition.” Any future claim must establish
the conjunction of budgeted client locality, complete-context action utility,
and measurable reader benefit.

## 4. Engineering and reproducibility

- Formal run code and V18 protocol/reporting utilities were synchronized to
  GitHub. Key commits: `fac9f62` (formal aggregation fix/start point),
  `2ae03dd` (V18 preregistration), `6778dbe` (integrity contract fix), and
  `7926d54` (recorded decision report).
- The reportable V18 artifacts are:
  - `V7-HP-PAPER/v18_opportunity_gated_fedaction/reports/checkpoint_a_result_report.md`
  - `V7-HP-PAPER/v18_opportunity_gated_fedaction/literature/decision_object_matrix.md`
  - `V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a/checkpoint_a_integrity.json`
  - `V7-HP-PAPER/v18_opportunity_gated_fedaction/checkpoint_a/checkpoint_a_all_units.csv`

## 5. Risks, decisions, and next-week plan

### Main risk identified

Cross-client evidence dispersion is not sufficient: under Bc<=3, candidate
availability and reader-beneficial composition are separated by a large gap.
Training a federated composer now would risk optimizing an Oracle action space
that the real reader does not consistently reward.

### Decision

Follow the preregistered fail branch: **return to reader-aligned selective
retriever-block upload**. Preserve V17 as a rigorous federated opportunity and
systems-analysis artifact rather than continuing to add complexity after a
failed gate.

### Planned work

1. Consolidate the V7-HP selective-upload line with the improved V17 source
   routing, communication accounting, and two-reader evaluation contract.
2. Define a new, independently frozen development protocol before considering
   a targeted high-dispersion slow path; its subgroup detector must be
   inference-safe and cannot use gold evidence.
3. Turn the decision-object literature matrix into a baseline contract table
   for the next selective-upload experiment and paper narrative.
4. Archive the V17/V18 artifacts and prepare a concise group-discussion slide
   explaining why a clean negative Go/No-Go result is preferable to a
   post-hoc rescue experiment.

## 6. Assistance needed

- Advice on whether the next paper should foreground the reader-aligned
  selective-upload direction directly, or first develop a targeted
  high-dispersion opportunity analysis as a separate systems/measurement
  contribution.
- Feedback on which venue best fits a rigorously negative federated
  opportunity-analysis result versus a future positive selective-upload method.
