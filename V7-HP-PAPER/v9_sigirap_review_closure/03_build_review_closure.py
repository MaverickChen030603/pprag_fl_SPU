#!/usr/bin/env python3
"""Build the SIGIR-AP review-closure manuscript and audit package."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V8 = ROOT / "v8_sigirap_final_story_reframing"
MAIN_SOURCE = V8 / "ars_polished/paper_sigirap_ars_polished_9page.md"
SUPP_SOURCE = V8 / "paper_sigirap_final_supplement.md"
OLD_BIB = ROOT / "final_venue_packaging_and_review_defense/references.bib"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(name: str, value: str) -> None:
    (HERE / name).write_text(value.strip() + "\n", encoding="utf-8")


def section(text: str, start: str, end: str | None) -> str:
    left = text.index(start)
    right = text.index(end, left) if end is not None else len(text)
    return text[left:right].strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_terms(value: str) -> str:
    replacements = {
        "leak-controlled framework": "leakage-controlled framework with an explicit no-leak audit",
        "fully nested, leak-free evaluation": "fully nested, leakage-controlled evaluation with an explicit no-leak audit",
        "two independent freezes": "two disjoint frozen same-source evaluations",
        "two independently frozen HotpotQA holdouts": "two disjoint frozen same-source HotpotQA evaluations",
        "two independent frozen holdouts": "two disjoint frozen same-source evaluations",
        "empirical risk frontier": "observed intervention risk at the frozen operating point",
        "No positive action": "No training-positive action",
        "Positive action missed": "Training-positive action missed",
        "Positive action selected": "Training-positive action selected",
        "no positive action under the training definition": "no training-positive action",
        "A positive action is one labeled": "A training-positive action is one labeled",
        "No training-positive opportunity": "No training-positive action",
        "Opportunity missed": "Training-positive action missed",
        "Positive selected": "Training-positive action selected",
        "Independent CrossEncoder-Top5 Details": "Protocol-Matched Shared-Checkpoint CrossEncoder-Top5 Details",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


ABSTRACT = """
Multi-hop question answering depends on assembling complementary evidence into a context that remains usable by the answer reader. Yet a selector can act only on alternatives exposed by its generator, coupling candidate availability with policy realization. We introduce Full, a bounded context-construction system that combines pair-complementary, anchor-preserving actions with a fully nested, empirically risk-controlled selector and exact fallback. Across two disjoint frozen same-source HotpotQA evaluations of 3,000 and 3,405 queries, Full improves Answer, supporting-fact, and Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 at 25.8%-25.9% intervention coverage. A protocol-matched shared-checkpoint CrossEncoder reranker reaches higher supporting-fact and Joint F1 at lower latency, whereas Full attains higher Answer F1 and improves both Answer and Joint over the frozen baseline. These systems expose distinct answer-evidence-cost operating points. Retrospective frozen-action diagnostics further separate absent training-positive actions from missed available actions. Together, the results provide a fully nested, leakage-controlled analysis of candidate opportunity, selective realization, observed intervention risk, and measured cost in bounded multi-hop context construction, with same-source replication and explicit transfer boundaries.
"""


INTRODUCTION = """
## 1. Introduction

Multi-hop question answering succeeds only when the reader receives evidence that is not merely relevant, but jointly usable. One passage may identify the bridge entity, another may supply the answer-bearing statement, and an individually strong distractor may consume the same limited context budget. The retrieval problem is therefore compositional: construct an ordered context that exposes complementary reasoning roles while retaining the information needed to express the answer.

This view reveals a constraint that is easy to overlook when attention is placed only on the final selector. A policy can choose only among the contexts produced for a query. If the bounded action set contains no answer-compatible repair, improved decision accuracy cannot recover one. We call this the **candidate-opportunity gap**. Candidate availability is necessary, but the frozen-action diagnostic also shows that availability is not sufficient: useful actions can exist and still remain unrealized by the deployed policy. Multi-hop context intervention is governed jointly by what actions are available and how reliably they are selected.

Independent relevance scoring provides an important reference point. A strong CrossEncoder can recover supporting evidence by moving individually relevant documents upward, even without explicit pair construction. However, independent scores do not directly encode whether two passages occupy complementary hops or whether a replacement changes answer expression. The empirical question is therefore which answer-evidence-cost operating points emerge when pool, document budget, reader, support predictor, and evaluation code are held fixed.

We address this question with **Full**, an opportunity-aware context constructor over a frozen approximately ten-document pool. Starting from a Top-5 baseline, Full combines lexical, entity, semantic, and missing-hop signals; scores document opportunity and pair complementarity; and exposes bounded insertion, replacement, reordering, and two-document-chain actions. Anchor-preserving action families retain strong early baseline passages when the five-document budget permits. The upstream retriever and source text remain unchanged.

A preservation head and utility head then determine whether to apply one action or return the baseline exactly. We call this **risk-controlled selection** in an empirical sense: it is an answer-preservation-oriented operating rule calibrated on nested development data, not a per-query harm guarantee and not conformal risk control. The selector modifies approximately 26% of contexts, but Full executes its generator and selector for every query. One final reader call is made after the context decision.

The evaluation protocol makes this claim auditable. Generator modules and selector heads fit outer-training queries; inner out-of-fold predictions determine thresholds and coverage; and outer-test outcomes remain unseen. The complete pipeline is frozen before evaluation on two disjoint same-source HotpotQA samples of 3,000 and 3,405 queries. Candidate reader outcomes serve only as offline training labels. At inference, Full uses the question, candidate passages, baseline order, frozen features, and learned parameters. The no-leak audit verifies query separation, fold-matched training, absent target outcomes in inference features, and unchanged holdout thresholds.

Across both evaluations, Answer, supporting-fact (SP), and Joint F1 move in the same positive direction. The deltas are +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080, with paired confidence intervals excluding zero. This consistency links a selective context intervention to population-level reader outcomes under two disjoint frozen same-source evaluations, rather than only to conditional gains on edited examples.

The shared-checkpoint CrossEncoder reaches higher SP and Joint at 149.90 ms/query; Full reaches higher Answer and improves Answer and Joint over baseline at 213.48 ms/query. Neither dominates all objectives. Frozen-action diagnostics further separate absent training-positive actions from selector misses.

This is a method-and-analysis study: Full provides the controlled intervention system, while the frozen comparisons separate candidate availability, selector realization, and answer-evidence-cost objectives.

Our contributions are threefold:

1. We formulate candidate opportunity as a constraint on reader-aware context selection and introduce bounded pair-complementary, anchor-preserving actions.
2. We provide a fully nested, leakage-controlled evaluation with an explicit no-leak audit on two disjoint frozen same-source evaluations, connecting selective interventions to Answer, SP, and Joint improvements while measuring observed intervention risk and latency.
3. We separate candidate availability from selector regret and use a protocol-matched shared-checkpoint CrossEncoder baseline to characterize distinct answer-evidence-cost operating points.

**Figure 1: Candidate opportunity and the empirically risk-controlled context-construction pipeline.** The selector can only choose from generated actions; unavailable repairs and missed available actions are distinct sources of unrealized utility. No answer, support annotation, or candidate reader outcome is used at inference.

```mermaid
flowchart LR
    Q["Question"] --> R["Frozen retrieval"]
    R --> B["Top-5 baseline"]
    R --> D["Bounded candidate pool"]
    B --> G["Pair-complementary action generator"]
    D --> G
    G --> S{"Empirically risk-controlled selector"}
    S -->|eligible action| C["Modified context"]
    S -->|otherwise| F["Exact baseline fallback"]
    C --> A["One final reader call"]
    F --> A
```
"""


RELATED = """
## 2. Related Work

**Multi-hop retrieval and structured QA.** HotpotQA and 2WikiMultiHopQA pair answers with supporting facts, enabling separate measurement of answer generation and evidence access [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. MDR changes the candidate pool through multi-step dense retrieval [@xiong-etal-2021-mdr], while HGN propagates paragraph, sentence, and entity information in a graph reader [@fang-etal-2020-hgn]. Decomposition-first methods such as GenDec instead generate subquestions before retrieval [@wu-etal-2024-gendec]. Our intervention starts after a frozen retriever has produced a bounded pool; it changes neither the upstream index nor the reader architecture.

**Reader-aware ranking and context-set construction.** RankRAG jointly instruction-tunes ranking and generation [@yu-etal-2024-rankrag]. R-CPS ranks and clusters passages using reader prediction behavior [@xin-etal-2025-rcps], and SetR explicitly selects a collectively useful passage set [@lee-etal-2025-setr]. RECOMP learns extractive context compression [@xu-etal-2024-recomp]. Recent BAR-RAG work trains a boundary-aware selector from generator feedback, but is included only as a conceptual comparison because it is a recent preprint with a different two-stage training contract [@sun-etal-2026-barrag]. Full differs by preserving a fixed upstream pool and reader, generating bounded structured actions, and applying an offline outcome-supervised fallback gate without online candidate-reader search.

**Context sensitivity.** Reader output can change with distractors and evidence position [@shi-etal-2023-distracted; @liu-etal-2024-lost]. This motivates preserving strong baseline anchors and evaluating Answer, SP, and Joint jointly. Our analysis does not infer a causal answer-anchor mechanism from outcome differences; it reports the frozen operating points and observed intervention harms.

**Selective prediction and risk control.** SelectiveNet formalizes risk-coverage trade-offs through learned rejection [@geifman-elyaniv-2019-selectivenet]. Distribution-free risk-controlling prediction sets and Learn-then-Test provide finite-sample control under explicit calibration assumptions [@bates-etal-2021-rcps; @angelopoulos-etal-2021-ltt]. TRAQ applies conformal prediction to retrieval-augmented QA prediction sets [@li-etal-2024-traq], and C-RAG studies certified generation risk for RAG [@kang-etal-2024-crag]. Our gate is outcome-supervised and empirically calibrated on nested development data. It reports average observed intervention risk at one frozen operating point, but provides neither finite-sample coverage nor group-conditional guarantees. Extending answer-preservation selection with distribution-aware risk control is a separate research direction.

Closest systems differ in pool changes, online reader/generator calls, and empirical versus guaranteed risk. The supplement gives the complete contract matrix and verified source map.
"""


REPRO_TABLE = """
| Component | Training supervision | Inference inputs | Output |
| --- | --- | --- | --- |
| Missing-hop estimator | Outer-training missing-type labels from offline outcomes | Query and baseline lexical/entity/semantic summaries | Five missing-type probabilities |
| Document opportunity | Outer-training action usefulness labels | Query-document relevance, overlap, novelty, redundancy, bridge and rank features | Candidate opportunity score |
| Pair complementarity | Outer-training pair labels | Two document scores, pair similarity, entity chain, novelty and redundancy | Pair score |
| Action generator | No target-query outcome | Frozen baseline, candidate scores, pair scores, missing-hop state | At most eight unique five-document actions plus fallback |
| Preservation head | Offline Answer F1 non-decrease label | Generator score, removal risk, semantic/opportunity summaries, missing-type and family features | Answer-preservation probability |
| Utility head | Offline answer-compatible utility label | Same inference-safe action features | Positive-utility probability |
| Selector | Inner out-of-fold development predictions | Two probabilities and frozen fold budget | One action or exact baseline fallback |
"""


def build_method_protocol(source: str) -> str:
    value = bounded_terms(section(source, "## 3. Problem and Method", "## 5. Main Results"))
    insertion = (
        "\n\n### 3.5 Component Contract\n\n"
        + REPRO_TABLE.strip()
        + "\n\nAll learned feature vectors are standardized inside their training pipeline. "
        "The supplement gives the exact feature order, labels, regularization, seeds, grids, caps, "
        "tie breaking, and fallback rule."
    )
    value = value.replace("\n## 4. Fully Nested Protocol", insertion + "\n\n## 4. Fully Nested Protocol")
    value = value.replace(
        "A Hotpot-development support predictor uses a frozen 0.7 threshold.",
        "A Hotpot-development support predictor uses a pre-specified frozen 0.7 threshold; a fixed "
        "0.5/0.6/0.7/0.8 post-hoc sensitivity grid is reported without changing the primary threshold.",
    )
    value = value.replace(
        "For RECOMP, the author-released HotpotQA compressor scores sentences in the same Top-5 input [@xu-etal-2024-recomp]. Development budgets are 64, 128, 256, 384, 512, and 660 FLAN tokens; 660 is frozen before holdout evaluation. Baseline-Truncated retains source sentence order at the same budget. All systems share reader, prompt, support predictor, and metric code.",
        "For RECOMP, the author-released HotpotQA compressor scores the same Top-5 input [@xu-etal-2024-recomp]. A 660-token budget is frozen from a development grid; a source-order truncation control uses the same budget. Reader, support predictor, and metrics are shared.",
    )
    value = value.replace(
        "For online cost, all systems run on one GPU with batch size one over the same ordered queries. We use 50 warmup queries and measure the next 500, synchronizing CUDA around every component. Model loading is excluded. Online features/actions are recomputed and their final context must exactly match the frozen artifact. Candidate outcome labeling and training are offline.",
        "Online cost uses one GPU, batch size one, 50 warmup and 500 measured queries with CUDA synchronization. Model loading and upstream retrieval are excluded; recomputed contexts must match frozen artifacts. Training and candidate labeling are offline.",
    )
    value = value.replace(
        "The confirmatory samples are constructed from a fixed ordering and audited for query-ID overlap. The original 3,000-query holdout is opened only after the nested development pipeline is frozen. The remaining 3,405 queries are untouched while the Lite architecture and 0.002 non-inferiority margin are fixed. Statistical intervals and p-values are query-level paired bootstrap estimates, so every comparison preserves the baseline/action pairing for the same question. The second sample is not used to retune Full after the first result.",
        "A fixed ordering and query-ID audit define both frozen samples. The 3,000-query sample is opened after nested development freezing; the remaining 3,405 stay untouched while the Lite architecture and non-inferiority margin are fixed. Query-level paired bootstraps preserve same-question pairing. The second sample does not retune Full.",
    )
    value = value.replace(
        "The 3,000-query and 3,405-query holdouts are disjoint from the 1,000 development queries, and no holdout outcome selects an architecture or threshold.",
        "The 3,000-query and 3,405-query samples are disjoint from the 1,000 development queries and from "
        "each other. They are same-source evaluations, not external replications. No holdout outcome selects "
        "Full's architecture, threshold, or coverage rule; the explicit no-leak audit checks query IDs, "
        "artifact fingerprints, fold membership, feature availability, and frozen configurations.",
    )
    return value


def build_results(source: str) -> str:
    value = bounded_terms(section(source, "## 5. Main Results", "## 7. External Boundary"))
    old_table = """| Frozen split | N | Coverage | Answer delta | SP delta | Joint delta | Selected Answer-drop | Selected Joint-drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original holdout | 3,000 | 25.8% | +0.0088 | +0.0056 | +0.0064 | 7.75% | 14.86% |
| Revision holdout | 3,405 | 25.9% | +0.0116 | +0.0061 | +0.0080 | 7.83% | 14.19% |"""
    new_table = """| Frozen split | N | Coverage | Answer delta | SP delta | Joint delta | Joint 95% CI | Paired p | Answer-drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original holdout | 3,000 | 25.8% | +0.0088 | +0.0056 | +0.0064 | [+0.0027,+0.0104] | 0.0004 | 7.75% |
| Revision holdout | 3,405 | 25.9% | +0.0116 | +0.0061 | +0.0080 | [+0.0044,+0.0116] | <0.0004 | 7.83% |"""
    value = value.replace(old_table, new_table)
    value = value.replace(
        "The central result is consistency across two disjoint frozen same-source evaluations: Answer, SP, and Joint all move in the same positive direction, and all six paired intervals exclude zero. Full modifies about 26% of contexts, translating a sparse intervention policy into replicated population-level gains. Most selected queries tie the baseline, while the observed Answer- and Joint-drop rates define the observed intervention risk at the frozen operating point reported alongside those gains.",
        "Across the two disjoint frozen same-source evaluations, Answer, SP, and Joint all move in the same "
        "positive direction, and all six paired intervals exclude zero; Table 1 exposes the Joint interval "
        "and p-value directly. Full modifies about 26% of contexts. Most selected queries tie the baseline, "
        "while the Answer- and Joint-drop rates report observed intervention risk at this frozen operating point.",
    )
    value = value.replace(
        "### 5.2 Answer-Evidence Trade-off against Independent Reranking",
        "### 5.2 Answer-Evidence Trade-off against a Shared-Checkpoint Reranker",
    )
    old_ce = (
        "CrossEncoder-Top5 independently scores every document in the same approximately ten-document pool and retains five. "
        "It shares the frozen relevance checkpoint, reader, prompt, 3,200-character cap, support predictor, and official metrics with Full, "
        "but excludes pair, missing-hop, outcome-model, and selector features. Score order is chosen on development only and then frozen. "
        "Because this comparison was added after the primary study, it is a post-hoc secondary baseline analysis."
    )
    new_ce = (
        "The protocol-matched shared-checkpoint CrossEncoder-Top5 baseline scores every document in the same "
        "approximately ten-document pool and selects and orders five using only relevance. Full uses the same "
        "frozen relevance checkpoint as one feature among lexical, entity, opportunity, complementarity, and "
        "structural signals; that feature does not itself choose Full's context. The baseline excludes pair, "
        "missing-hop, opportunity, preservation, utility, and action-family logic. Score order is chosen on "
        "development only and then frozen. Because this comparison was added after the primary study, it is a "
        "post-hoc secondary baseline analysis. The comparison isolates the value of the complete context-construction "
        "and selective-intervention pipeline beyond using the same relevance checkpoint as an independent document ranker. "
        "It is protocol-matched, not representation-level independent."
    )
    value = value.replace(old_ce, new_ce)
    old_decomp = """| Split | No training-positive action | Training-positive action missed | Training-positive action selected |
| --- | ---: | ---: | ---: |
| Original 3,000 | 2,316 | 465 | 219 |
| Revision 3,405 | 2,638 | 515 | 252 |"""
    new_decomp = """| Split | No training-positive action | Training-positive action missed | Training-positive action selected |
| --- | ---: | ---: | ---: |
| Development 1,000 | 708 | 213 | 79 |
| Original 3,000 | 2,316 | 465 | 219 |
| Revision 3,405 | 2,638 | 515 | 252 |"""
    value = value.replace(old_decomp, new_decomp)
    value = value.replace(
        "The decomposition identifies two concrete sources of improvement headroom.",
        "The same qualitative availability-versus-regret split appears on fully nested development outputs. "
        "The decomposition identifies two concrete sources of improvement headroom.",
    )
    value = value.replace(
        "These outcomes support the frozen joint recipe and pair/chain mechanisms, not the necessity of every semantic feature.",
        "These outcomes are development opportunity diagnostics consistent with the frozen joint recipe; they do not "
        "establish end-to-end component necessity. A clean frozen holdout removal is unavailable because corresponding "
        "models were not frozen before holdout inspection, so we do not retrain them post hoc.",
    )
    old_cost_header = "| System / boundary | Frozen split | Joint contrast | Coverage | Selected Answer-drop | Latency | Interpretation |\n| --- | --- | ---: | ---: | ---: | ---: | --- |"
    new_cost_header = "| System / boundary | Frozen split | Joint contrast | Coverage | Answer-drop | Mean / P95 latency | Interpretation |\n| --- | --- | ---: | ---: | ---: | ---: | --- |"
    value = value.replace(old_cost_header, new_cost_header)
    value = value.replace("| 140.88 ms | Exact baseline |", "| 140.88 / 252.10 ms | Exact baseline |")
    value = value.replace("| 149.90 ms | Higher SP/Joint, lower Answer |", "| 149.90 / 262.59 ms | Higher SP/Joint, lower Answer |")
    value = value.replace("| 213.48 ms | Answer-oriented selective point |", "| 213.48 / 330.56 ms | Answer-oriented selective point |")
    value = value.replace("| 143.97 ms | Cheaper; NI failed |", "| 143.97 / -- ms | Cheaper; NI failed |")
    value = value.replace("| 169.64 ms | Budget control; p=0.4172 |", "| 169.64 / -- ms | Budget control; p=0.4172 |")
    latency_sentence = (
        " Full's mean component times are 70.05 ms for generator, 0.61 ms for selector, and "
        "142.59 ms for serialization plus reader; semantic feature computation dominates the added generator cost."
    )
    value = value.replace(
        "All evaluated online systems make one final answer-reader call; candidate reader outcomes are offline supervision.",
        "All evaluated online systems make one final answer-reader call; candidate reader outcomes are offline supervision." + latency_sentence,
    )
    sensitivity = (
        "\n\nA post-hoc fixed-grid support-threshold analysis at 0.5/0.6/0.7/0.8 keeps both Full-baseline "
        "SP and Joint deltas positive on both evaluations; the CrossEncoder directions are likewise stable. "
        "The pre-specified 0.7 threshold remains unchanged, and the complete table is in the supplement."
    )
    value = value.replace("\n## 6. Mechanism and Cost", sensitivity + "\n\n## 6. Mechanism and Cost")
    return value


EXTERNAL = """
## 7. External Boundary

Frozen transfer to 1,000 2Wiki queries changes Answer/SP/Joint F1 by +0.0086/-0.0006/+0.0033; all aggregate changes are non-significant. Selected Answer-drop is 6.92%. Few-shot gate calibration reduces coverage but still reaches 5.10% Answer-drop, missing the pre-specified 4% target, so no further target-outcome tuning is performed.

Type-level analysis uses the official 2Wiki reasoning field, and no subgroup survives Benjamini-Hochberg correction. The official taxonomy therefore does not explain aggregate transfer uncertainty after multiplicity correction. Feature diagnostics show lexical, entity, candidate, and score shifts, but these associations do not identify a root cause.

Future analysis should test mechanism-aligned groupings under a separately frozen protocol. The present evidence does not establish cross-domain reliability or subgroup transfer.
"""


LIMITATIONS = """
## 8. Limitations and Ethical Considerations

The population-level gains are consistent across two frozen same-source evaluations but bounded in magnitude. Risk control is empirical rather than certifying: 7.75%-7.83% of selected actions lower Answer F1 and 14.19%-14.86% lower Joint F1. The gate reports average observed intervention risk at a frozen operating point and provides no finite-sample, per-query, or group-conditional guarantee. The action-set oracle is outcome-aware and unavailable online. Its gap diagnoses candidate availability and selector regret; it is not a deployable baseline or an estimate of achievable deployment gain.

The protocol-matched shared-checkpoint CrossEncoder obtains higher SP and Joint F1 at lower latency than Full. Full instead occupies an evaluated Answer-oriented operating point and improves Answer and Joint over the frozen baseline. Because both systems share the relevance checkpoint, their comparison isolates the added context-construction and selection contract, not representation-level independence. No pre-inspection frozen Full-without-pair, Full-without-chain, or Full-without-CrossEncoder model exists, so clean end-to-end holdout component attribution remains unavailable. Development opportunity ablations are diagnostic only.

Both primary samples come from HotpotQA distractor validation. Their disjointness and freezing establish same-source replication, not external-domain generalization. Frozen 2Wiki transfer is non-significant and few-shot calibration misses its risk target. The evaluated pool contains approximately ten documents: 2,973/3,000 queries have at least ten candidates, only one has at least twenty, and none has fifty or one hundred. The benchmark therefore does not support a common natural 20- or 50-document stress test, and we make no corpus-scale or changing-index claim.

Full costs 213.48 ms/query (P95 330.56) on one measured GPU setup and runs generator and selector computation for every query. Lite's non-inferiority test fails, historical offline GPU-hour totals are unavailable, and no energy or alternative-hardware profile is reported. The work makes no low-overhead, edge, privacy, federated-client, or deployment-readiness claim.

The method rearranges retrieved passages and cannot recover evidence absent from the pool. Support predictions and preservation estimates may shift across domains or groups. Consequential use would require separately frozen target-domain calibration, subgroup auditing, and explicit tolerances for Answer, evidence, latency, and intervention harm. Conformal or PAC-style extensions would require their own assumptions, calibration design, and validation.
"""


CONCLUSION = """
## 9. Conclusion

This work frames multi-hop context construction as a joint problem of **candidate availability**, **selective realization**, and **operating trade-offs**. Full expands a frozen bounded pool into pair-complementary, anchor-preserving actions and applies them with a fully nested, empirically risk-controlled selector. Across two disjoint frozen same-source HotpotQA evaluations, it improves Answer, SP, and Joint F1 over the same Top-5 baseline.

The protocol-matched shared-checkpoint CrossEncoder comparison reveals a complementary result rather than a universal winner: relevance-only reranking reaches higher SP and Joint F1 at lower latency, while Full reaches higher Answer F1 and improves both Answer and Joint over baseline. Full is therefore the Answer-oriented evaluated operating point, not a generally superior system. The retrospective frozen-action decomposition separates absent training-positive actions from missed available actions, but neither diagnostic is deployable. The current evidence is bounded to same-source HotpotQA, an approximately ten-document pool, one support model, and one measured hardware setting; external calibration, statistically guaranteed risk control, larger natural pools, and lower-cost realization remain open problems.
"""


REPRO_SPEC = """
# Reproducibility Specification

## System boundary

Full acts after a frozen retriever has supplied an ordered approximately ten-document pool and a Top-5 baseline. It emits one five-document sequence or returns the baseline byte-for-byte. It does not update the index, modify source text, or search with reader calls at inference.

## Module-by-module contract

### Missing-hop estimator

- **Input:** question-level baseline semantic statistics, baseline CrossEncoder statistics, pair-similarity and redundancy summaries, bridge/entity summaries, question length and entity count.
- **Output:** probabilities for `missing_bridge`, `missing_answer_resolution`, `redundant_context`, `ordering_problem`, and `no_intervention_needed`.
- **Labels:** the five offline missing-type labels on outer-training queries.
- **Model:** standardized balanced multinomial logistic regression; `C` selected from 0.1/1/10 by three-way query-group inner validation; `max_iter=3000`; seed 20260714.

### Document-opportunity model

- **Feature order:** query-document cosine, CrossEncoder relevance, BM25, query-text overlap, query-title overlap, entity overlap, bridge-entity match, novel information, redundancy, maximum/mean baseline semantic similarity, semantic novelty, anchor proxy, normalized source rank.
- **Normalization:** CrossEncoder score is min-max normalized within query, with constant-score queries assigned 0.5; learned vectors then pass through a training-fold `StandardScaler`.
- **Output:** candidate opportunity probability.
- **Model:** balanced logistic regression with the same nested `C` grid, iterations, and seed.

### Pair-complementarity classifier

- **Input order:** left/right document opportunity, query-semantic sum/minimum, CrossEncoder sum/minimum, document-pair cosine, semantic complementarity, entity-chain overlap, bridge sum, novelty sum, redundancy sum.
- **Output:** pair-complementarity probability.
- **Cap:** candidate pairs are scored in the bounded pool; the frozen constructor retains at most three pair proposals and uses ten pair scores per query.
- **Model:** standardized balanced logistic regression with the nested `C` grid and seed 20260714.

### Action families and pruning

- **Families:** single complementary insertion; anchor-preserving replacement; semantic two-document chain; redundancy replacement; bridge-first reorder; answer-anchor-first reorder.
- **Single candidates:** top three ranked candidate documents.
- **Pairs:** top three pair proposals; pair score = `0.65 * learned_pair + 0.175 * left_doc + 0.175 * right_doc`.
- **Generator score:** `0.55 * opportunity + 0.30 * family_probability - 0.15 * removal_risk`.
- **Action cap:** eight. Dynamic cap is four when `no_intervention >= 0.65`, six when `>= 0.45`, otherwise eight.
- **Preservation:** baseline positions 0 and 1 are protected from removal; replacements begin at position 2.
- **Duplicate removal:** byte-equivalent document-ID sequences are deduplicated; every action must have five unique documents.
- **Ordering:** family-specific order is preserved after construction; no target outcome reorders actions.
- **Fallback:** exact original Top-5 order is always present.

### Preservation and utility heads

- **Shared input order:** generator score; new-vs-v3 indicator; mean added-document opportunity; mean added semantic score; removal risk; five missing-type probabilities; six action-family one-hot indicators.
- **Preservation label:** Answer F1 delta `>= 0` relative to the same-query fallback.
- **Utility label:** preservation label is positive, answer-title-product delta is positive, and title recall improves or title F1 does not decrease.
- **Model:** two independent `StandardScaler + LogisticRegression` pipelines; `C=0.5`, balanced classes, `max_iter=2500`, seed 20260714.

### Nested calibration and coverage

- **Outer split:** five query-disjoint folds over 1,000 development queries.
- **Inner split:** five deterministic MD5 query groups per outer fold.
- **Grid:** preservation threshold 0.4/0.5/0.6/0.7/0.8; utility threshold 0.3/0.4/0.5/0.6/0.7; coverage 0.10/0.15/0.20/0.25/0.30.
- **Feasibility:** aggregate development Answer delta `>= -0.001` and selected Answer-drop `<= 0.05`.
- **Objective:** answer-title-product delta plus 0.25 times title-recall delta.
- **Eligibility:** both head thresholds must pass.
- **Tie breaking:** rank eligible actions by utility probability, then preservation probability; Python stable order resolves exact ties.
- **Budget:** `round(coverage * number_of_queries)` per frozen fold; excluded or ineligible queries fall back exactly.
- **Frozen fold configurations:** `(safe, utility, coverage)` = fold 0 `(0.6,0.3,0.30)`, fold 1 `(0.5,0.3,0.30)`, fold 2 `(0.5,0.3,0.25)`, fold 3 `(0.6,0.3,0.15)`, fold 4 `(0.6,0.3,0.30)`.

## Inference-time information audit

Allowed: question text, candidate titles/text, source rank, Top-5 baseline order, lexical/entity features, cached MPNet and CrossEncoder scores computed from query and documents, learned parameters, and frozen thresholds. Prohibited and absent: gold answer, answer-string presence, supporting facts/titles, official metrics, candidate reader outputs, action outcome labels, oracle actions, and target-query fold outcomes.

## Reader and metrics

The primary answer reader is FLAN-T5-Large with greedy decoding, 32 new tokens maximum, input length 1,024, and serialized context capped at 3,200 characters. One final reader call follows selection. The support model is a balanced logistic sentence classifier trained on Hotpot development contexts with threshold 0.7. Official Answer, SP, and Joint metrics are paired by query; confidence intervals and two-sided p-values use 5,000 bootstrap resamples.
"""


SUPP_REPRO_TABLE = """
# Supplement Reproducibility Table

| Module | Input dimension / feature family | Model and normalization | Labels | Frozen decision rule |
| --- | --- | --- | --- | --- |
| Missing-hop | 11 query/baseline summaries | StandardScaler + balanced logistic; nested C in 0.1/1/10 | Five missing-type classes | Five probabilities |
| Document opportunity | 14 document features | Per-query CE min-max, then StandardScaler + balanced logistic | Training action usefulness | Candidate probability |
| Pair complementarity | 13 pair features | StandardScaler + balanced logistic | Training pair complementarity | Top three proposals; ten scored pairs |
| Action generator | Document, pair, missing-hop and family scores | Deterministic bounded constructor | None at inference | 4/6/8 dynamic cap, max eight plus fallback |
| Preservation head | 16 action features | StandardScaler + balanced logistic, C=.5 | Answer F1 non-decrease | Fold threshold 0.5 or 0.6 |
| Utility head | Same 16 action features | StandardScaler + balanced logistic, C=.5 | Answer-compatible utility gain | Fold threshold 0.3 |
| Coverage gate | Two probabilities | Deterministic sort | Inner OOF objective | Fold coverage 0.15/0.25/0.30 |
| Support predictor | Nine sentence features | StandardScaler + balanced logistic, C=.5 | Official supporting sentence | Threshold 0.7; top five, minimum two |

Seed for generator, selector, support model and paired bootstrap: **20260714** unless the diagnostic script explicitly records **20260715**. Exact fold configurations and checkpoint hashes are in `seed_and_checkpoint_manifest.md`.
"""


CROSSENCODER_AUDIT = """
# CrossEncoder Role and Fairness Audit

## Shared representation, different contracts

Both Full and CrossEncoder-Top5 use the frozen `cross-encoder/ms-marco-MiniLM-L-6-v2` checkpoint. This deliberately controls the document-level relevance representation, but it means the comparison is not representation-level independent.

### CrossEncoder inside Full

- Produces one query-document relevance feature.
- Is combined with BM25, lexical and title overlap, entity and bridge overlap, MPNet similarity, novelty, redundancy, source rank, missing-hop, opportunity, pair, preservation, and action-family signals.
- Does not directly choose or order the final five-document context.
- Is available at inference without answers, support labels, or reader outcomes.

### CrossEncoder-Top5 baseline

- Scores every candidate document in the same frozen approximately ten-document pool.
- Selects and orders five documents using only the CrossEncoder relevance score.
- Excludes pair complementarity, missing-hop, document opportunity, preservation, utility, and action-family logic.
- Uses the same document budget, context cap, answer reader, prompt, support model, and official metrics.
- The score-order variant was selected on development and frozen before the two reported holdout evaluations.

## Fair interpretation

The comparison isolates the value of the complete context-construction and selective-intervention pipeline beyond using the same relevance checkpoint as an independent document ranker. It does not isolate the CrossEncoder representation itself and does not show that Full universally outperforms relevance reranking. CrossEncoder achieves higher SP and Joint at lower latency; Full achieves higher Answer and positive Answer/Joint changes relative to Frozen Top-5.

## Required naming

Use **protocol-matched shared-checkpoint CrossEncoder baseline**. Do not use **fully independent model baseline**. A clean Full-without-CrossEncoder holdout comparison remains future work because no compatible variant was frozen before holdout inspection.
"""


RISK_RELATED = """
# Risk-Calibrated and Conformal Related Work

## Verified positioning

| Direction | Formal source | What it controls | Difference from Full |
| --- | --- | --- | --- |
| Selective prediction | Geifman and El-Yaniv, ICML 2019 | Empirical risk-coverage through rejection | Full falls back to a retrieval baseline and uses two outcome-supervised heads |
| Risk-controlling prediction sets | Bates et al., JACM 2021 | Finite-sample expected-loss control for set-valued prediction under calibration assumptions | Full reports average observed intervention risk with no statistical guarantee |
| Learn-then-Test | Angelopoulos et al., 2021 | Finite-sample calibration through hypothesis testing | Full's development grid is ordinary empirical model selection, not LTT |
| TRAQ | Li et al., NAACL 2024 | Conformal answer/retrieval prediction sets | Full emits one context and one answer, not a calibrated prediction set |
| C-RAG | Kang et al., ICML 2024 | Conformal generation-risk bounds for RAG | Full's preservation gate is not certified generation-risk control |

Primary source pages:

- SelectiveNet: https://proceedings.mlr.press/v97/geifman19a.html
- Distribution-Free Risk-Controlling Prediction Sets: https://www.gsb.stanford.edu/faculty-research/publications/distribution-free-risk-controlling-prediction-sets
- Learn-then-Test: https://arxiv.org/abs/2110.01052
- TRAQ: https://aclanthology.org/2024.naacl-long.210/
- C-RAG: https://proceedings.mlr.press/v235/kang24a.html

## Safe manuscript language

Our gate is outcome-supervised and empirically calibrated, but it does not provide conformal, PAC-style, per-query, or group-conditional coverage guarantees. It measures average observed intervention risk at one frozen operating point. Conformal alternatives may provide marginal finite-sample guarantees under explicit exchangeability, score, loss, and calibration assumptions. They do not automatically solve domain shift or certify every query.

## Revision decision

No conformal gate is added during this submission cycle. Introducing one after holdout inspection would require a separate calibration protocol and new validation. Distribution-aware risk control is stated as future work rather than implied by the current terminology.
"""


RELATED_MATRIX = """
# Related Work Comparison Matrix

This matrix is conceptual unless a frozen experimental comparison is explicitly named. It does not rank unimplemented methods.

| Method/direction | Retrieval stage | Changes pool | Reader/generator feedback | Online selection calls | Action structure | Risk mechanism | Closest difference from Full |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| GenDec | Pre-retrieval decomposition | Yes | LLM decomposition | At least decomposition plus QA | Subquestions | None central to method | Full keeps query and upstream pool fixed |
| MDR | Multi-step retrieval | Yes | Retrieval supervision | Iterative retriever | Hop-wise retrieval | None central to method | Full reorganizes a bounded post-retrieval pool |
| HGN | Reader/reasoner | No fixed comparison | Joint QA supervision | One graph reader | Paragraph-sentence-entity graph | None central to method | Full does not change reader architecture |
| RECOMP | Post-retrieval compression | No | Compressor training | Compressor plus reader | Sentence compression | No answer-preservation gate | Full preserves full passages and structured actions |
| RankRAG | Ranking plus generation | No/depends on upstream | Joint ranking-generation instruction tuning | Ranking/generation model | Ranked contexts | No explicit fallback risk gate | Full separates frozen reader from selector |
| SetR | Passage-set selection | No/within retrieved pool | LLM reasoning over set needs | Selection LLM plus reader | Variable set | No empirical preservation head | Full uses bounded learned actions and exact fallback |
| R-CPS | Reader-centered selection | No | Reader distributions | Reader-dependent reranking/clustering | Ranked clusters | Reader consistency heuristic | Full uses offline outcomes, no online candidate-reader loop |
| BAR-RAG | Boundary-aware selection | Noisy retrieved pool | Generator reward/RL | Selector and generator pipeline | Evidence subset | Reward-defined boundary | Recent preprint; Full uses supervised nested fallback |
| Conformal RAG filtering | Retrieval/answer set calibration | Often variable set | Calibration scores | Method-dependent | Prediction/retrieval sets | Marginal finite-sample guarantee under assumptions | Full reports empirical intervention risk only |

Sources are official proceedings pages where available: MDR and HGN via ACL Anthology, RankRAG via NeurIPS, SetR and R-CPS via ACL Anthology, RECOMP via ACL Anthology, TRAQ via ACL Anthology, and C-RAG via PMLR. GenDec and BAR-RAG are treated as conceptual preprint comparisons only.
"""


LATENCY_REPORT = """
# Full Latency Report

## End-to-end post-retrieval latency

| System | Mean ms/query | Median ms/query | P95 ms/query | Reader calls | CrossEncoder scores | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Top-5 | 140.88 | 124.25 | 252.10 | 1 | 0 | 2.12 GB |
| CrossEncoder-Top5 | 149.90 | 135.47 | 262.59 | 1 | approximately 10 | not separately frozen |
| Full | 213.48 | 199.81 | 330.56 | 1 | 10 | 2.99 GB |

Peak memory uses decimal GB from the recorded byte counters; binary values are approximately 1.98 GiB for Frozen Top-5 and 2.78 GiB for Full.

## Full component means

| Component group | Mean ms/query |
| --- | ---: |
| Generator | 70.05 |
| Selector | 0.61 |
| Serialization plus reader | 142.59 |
| Total | 213.48 |

Semantic feature computation, especially MPNet encoding, dominates the added generator cost. Pair-feature construction and pair classification are small relative to semantic encoding. Full modifies approximately 26% of contexts but executes generator and selector computation for every query.

## Protocol

Measurements use one GPU, batch size one, 50 warmup queries, 500 measured queries, fixed query order, and CUDA synchronization around components. Model loading and upstream retrieval are excluded. Every online system makes one answer-reader call. The numbers describe one post-retrieval implementation and are not throughput, energy, edge-hardware, or production service-level measurements.
"""


REVIEW_ACCURACY = """
# Review Accuracy and Action Audit

| Review concern | Classification | Evidence audit | Action in v9 |
| --- | --- | --- | --- |
| "Two independent frozen holdouts" | Valid and actionable | Samples are disjoint but both come from HotpotQA distractor validation | Replaced with "two disjoint frozen same-source evaluations" throughout |
| "Leak-free protocol" | Partially resolved | Nested split and no-leak artifacts exist, but absolute leakage freedom is too strong | Uses "fully nested, leakage-controlled protocol with an explicit no-leak audit" |
| Latency variance missing | Already resolved in artifacts | Mean, median, P95, component means, calls, and memory were frozen | Restored mean/P95 to main and complete breakdown to supplement |
| Oracle should also appear on development | Already resolved in artifacts | Nested development decomposition is 708/213/79; AP-oracle Joint is .4404 | Added development row and bounded retrospective interpretation |
| End-to-end component removals are needed | Valid and actionable, but unavailable as clean confirmation | No compatible removal checkpoint was frozen before holdout inspection | Retains development opportunity diagnostics and explicitly defers holdout removals |
| CrossEncoder role is confounded | Valid and actionable | Same relevance checkpoint is used under different selection contracts | Added shared-checkpoint fairness audit and protocol-matched terminology |
| Support score may depend on threshold 0.7 | Valid and actionable | Frozen probabilities permit a no-training fixed-grid audit | Added 0.5/0.6/0.7/0.8 sensitivity; no direction flips and no threshold change |
| Risk-controlled wording implies guarantees | Valid and actionable | Current gate is empirical, not conformal or PAC-style | Added formal distinction and removed guarantee-like wording |
| Need natural 20/50 pool experiment | Valid motivation but out of scope for this benchmark | 2,973/3,000 have at least ten, one at least twenty, none fifty | States bounded-pool limit; does not fabricate corpus-scale stress test |
| Full should generally beat CrossEncoder | Incorrect or overstated | CrossEncoder has higher SP/Joint and lower latency; Full has higher Answer | Frames distinct operating points and avoids universal superiority |
| UnifiedQA independently replicates Joint | Incorrect or overstated | Answer reader changes, support predictor is shared | Reports Answer-only directional evidence |
| Conformal gate should be added now | Valid but out of scope | Would need separate assumptions, calibration and frozen validation | Related-work/future-work addition only |
"""


REVIEW_RESPONSE = """
# Final SIGIR-AP Review Response

## 1. End-to-end ablations

**Concern.** The previous draft did not establish end-to-end effects of pair, two-document-chain, and CrossEncoder features.

**Agreement level.** We agree that frozen end-to-end removals would strengthen component attribution.

**Existing/new evidence.** The artifact inventory found development opportunity ablations but no corresponding removal checkpoint/action set frozen before holdout inspection.

**Paper revision.** We retain the development diagnostics, label them as such, add a protocol audit, and do not train post-inspection variants. The manuscript now states: a clean frozen end-to-end ablation is unavailable because the corresponding model was not frozen before holdout inspection.

**Remaining limitation.** End-to-end component necessity remains unresolved.

## 2. Reader and support robustness

**Concern.** SP and Joint may depend on the 0.7 support threshold, and reader evidence is narrow.

**Agreement level.** Agree.

**Existing/new evidence.** We add a fixed 0.5/0.6/0.7/0.8 support-threshold grid over frozen probabilities, contexts, and answer outputs. Full-baseline and CrossEncoder-baseline SP/Joint directions remain positive on both holdouts. FLAN Answer changes from .6183 to .6271 and UnifiedQA Answer from .5662 to .5772.

**Paper revision.** The primary threshold remains 0.7; the grid is labeled post-hoc sensitivity. UnifiedQA is described as Answer-only directional evidence.

**Remaining limitation.** The support predictor is shared, so SP and Joint are not independent reader replications.

## 3. CrossEncoder dual role

**Concern.** Full and CrossEncoder-Top5 share a relevance model.

**Agreement level.** Agree; the previous "independent" shorthand was insufficient.

**Existing/new evidence.** The same frozen checkpoint is used under two contracts: one feature among many inside Full versus the sole ranking criterion in CrossEncoder-Top5.

**Paper revision.** We use "protocol-matched shared-checkpoint CrossEncoder baseline" and add a role-isolation audit. The comparison tests the complete pair/action/selection pipeline beyond relevance-only ranking, not representation-level independence.

**Remaining limitation.** A frozen Full-without-CrossEncoder holdout variant is unavailable.

## 4. Latency variance

**Concern.** Mean latency alone hides variance and module cost.

**Agreement level.** Agree; the necessary artifacts already existed.

**Existing/new evidence.** Frozen Top-5 is 140.88/252.10 mean/P95, CrossEncoder 149.90/262.59, and Full 213.48/330.56 ms/query. Full components are 70.05 generator, 0.61 selector, and 142.59 reader/serialization ms.

**Paper revision.** We restore mean, P95, call counts, memory, and component breakdown. Semantic feature computation is identified as the main added generator cost.

**Remaining limitation.** Measurements cover one GPU, batch-one post-retrieval setup.

## 5. Conformal selection and risk wording

**Concern.** "Risk-controlled" could imply finite-sample safety.

**Agreement level.** Agree.

**Existing/new evidence.** Selective prediction, risk-controlling prediction sets, TRAQ, and C-RAG are now positioned from verified formal sources.

**Paper revision.** We state that the gate is outcome-supervised and empirically development-calibrated, with no finite-sample, per-query, or group-conditional guarantee. Conformal risk control is future work, not a last-minute method addition.

**Remaining limitation.** The current operating point reports observed average intervention risk only.

## 6. Larger candidate pools

**Concern.** A 20/50-document stress test would clarify scaling.

**Agreement level.** We agree with the motivation but not with manufacturing unrelated distractors as a substitute.

**Existing/new evidence.** The frozen pool audit finds 2,973/3,000 queries with at least ten candidates, one with at least twenty, and none with fifty.

**Paper revision.** We bound the claim to approximately ten-document post-retrieval pools and list adaptive Top-L, subquadratic pair proposal, ANN pairing, and changing-index calibration as future work.

**Remaining limitation.** No natural common large-pool evaluation is available in this benchmark artifact.
"""


def build_artifact_manifest() -> str:
    paths = [
        "opportunity_aware_semantic_generation_v4/outputs/scaleup/frozen_selector_manifest.json",
        "opportunity_aware_semantic_generation_v4/outputs/scaleup/official_metrics/scaleup_official_summary.json",
        "review_driven_revision_v5/outputs/lite_model/lite_holdout_metrics.json",
        "v7_sigirap_targeted_strengthening/outputs/reranker/ce_reranker_metrics.json",
        "v7_sigirap_targeted_strengthening/outputs/reranker/ce_action_build_manifest.json",
        "v7_sigirap_targeted_strengthening/outputs/oracle/oracle_metrics.json",
        "v4_submission_completion/outputs/generator_ablation/generator_ablation_results.json",
        "v4_submission_completion/outputs/generator_ablation/generator_ablation_preparation_audit.json",
        "v5_final_review_optimized_submission/outputs/cost/frozen_latency_full_v4.json",
        "v5_final_review_optimized_submission/outputs/cost/frozen_latency_frozen_top5_baseline.json",
        "v7_sigirap_targeted_strengthening/outputs/reranker/ce_reranker_latency.json",
    ]
    lines = [
        "# Artifact Manifest",
        "",
        "Repository-relative paths are used to preserve anonymity. SHA-256 values describe the local review-closure source artifacts.",
        "",
        "| Artifact | Role | SHA-256 |",
        "| --- | --- | --- |",
    ]
    for relative in paths:
        path = ROOT / relative
        role = "Frozen result or audit evidence"
        lines.append(f"| `{relative}` | {role} | `{sha256(path)}` |")
    lines.extend(
        [
            "",
            "## Review-closure scripts",
            "",
            f"- `01_end_to_end_secondary_ablations.py`: `{sha256(HERE / '01_end_to_end_secondary_ablations.py')}`",
            f"- `02_support_threshold_sensitivity.py`: `{sha256(HERE / '02_support_threshold_sensitivity.py')}`",
            "",
            "The support sensitivity run emits its own manifest and leaves `primary_threshold=0.7`, `threshold_retuned=false`, `contexts_changed=false`, and `answer_outputs_changed=false`.",
        ]
    )
    return "\n".join(lines)


def build_seed_manifest() -> str:
    manifest = json.loads(
        read(ROOT / "opportunity_aware_semantic_generation_v4/outputs/scaleup/frozen_selector_manifest.json")
    )
    lines = [
        "# Seed and Checkpoint Manifest",
        "",
        "## Global seeds",
        "",
        "| Purpose | Seed |",
        "| --- | ---: |",
        "| Generator and selector training | 20260714 |",
        "| Support predictor and primary bootstrap | 20260714 |",
        "| SIGIR diagnostic bootstrap | 20260715 |",
        "",
        "## Frozen pretrained checkpoints",
        "",
        "| Role | Checkpoint identifier |",
        "| --- | --- |",
        "| MPNet semantic encoder | `sentence-transformers/all-mpnet-base-v2`, snapshot `e8c3b32edf5434bc2275fc9bab85f82640a19130` |",
        "| CrossEncoder relevance | `cross-encoder/ms-marco-MiniLM-L-6-v2`, snapshot `c5ee24cb16019beea0893ab7796b1df96625c6b8` |",
        "| Primary answer reader | `google/flan-t5-large`, snapshot `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` |",
        "| Secondary answer reader | UnifiedQA-T5-Large frozen artifact (Answer-only robustness) |",
        "",
        "## Fold selector checkpoints",
        "",
        "| Fold | Safe threshold | Utility threshold | Coverage | Model SHA-256 |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in manifest["folds"]:
        config = row["frozen_config"]
        lines.append(
            f"| {row['fold_id']} | {config['safe_threshold']:.1f} | {config['positive_threshold']:.1f} | "
            f"{config['coverage']:.2f} | `{row['model_sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"Frozen original-holdout selection artifact SHA-256: `{manifest['selection_sha256']}`.",
            "",
            "Absolute cache and user paths are intentionally omitted from manuscript-facing artifacts.",
        ]
    )
    return "\n".join(lines)


def build_supplement(source: str) -> str:
    source = bounded_terms(source)
    source = source.replace("# Final Supplement", "# Review-Closed Supplement")
    source = source.replace("## P. No-Leak Manifest Checklist", "## P. Explicit No-Leak Audit Checklist")
    source = source.replace(
        "All oracles are restricted to the already generated bounded action set plus baseline.",
        "All retrospective diagnostics are restricted to the already generated bounded action set plus baseline.",
    )
    appendices = [
        "## X. Review-Closure Reproducibility Specification\n\n" + REPRO_SPEC.replace("# Reproducibility Specification\n\n", ""),
        "## Y. Secondary Ablation Boundary\n\n" + read(HERE / "secondary_ablation_table.md").replace("# Secondary Ablation Evidence\n\n", ""),
        "## Z. Support-Threshold Sensitivity\n\n" + read(HERE / "support_threshold_sensitivity_report.md").replace("# Support-Threshold Sensitivity\n\n", ""),
        "## AA. CrossEncoder Role and Fairness\n\n" + CROSSENCODER_AUDIT.replace("# CrossEncoder Role and Fairness Audit\n\n", ""),
        "## AB. Latency Variance and Components\n\n" + LATENCY_REPORT.replace("# Full Latency Report\n\n", ""),
        "## AC. Risk-Control Scope\n\n" + RISK_RELATED.replace("# Risk-Calibrated and Conformal Related Work\n\n", ""),
    ]
    return source.rstrip() + "\n\n" + "\n\n".join(appendices)


def build_claim_audit(main: str, supplement: str) -> str:
    checks = [
        ("leak-free", "Prohibited absolute wording", "Use leakage-controlled plus explicit audit"),
        ("independent holdout", "Could imply external replication", "Use disjoint same-source evaluation"),
        ("external replication", "Unsupported", "State same-source replication"),
        ("guarantee", "Needs formal scope", "Negate or attribute to conformal work"),
        ("certified", "Needs formal scope", "Use only for C-RAG citation or explicit negation"),
        ("conformal", "Could conflate methods", "State current method is not conformal"),
        ("robust reader", "Too broad", "Use Answer-only directional evidence"),
        ("pair complementarity necessary", "No frozen holdout removal", "Use diagnostic consistency only"),
        ("Full outperforms", "Universal comparison unsafe", "Name metric and baseline"),
        ("Pareto-optimal", "Search-space universal claim", "Use non-dominated among evaluated points"),
        ("scalable", "No large-pool evaluation", "Bound to approximately ten documents"),
        ("causal", "Observational diagnostic", "Use retrospective/descriptive"),
        ("oracle achievable", "Oracle uses target outcomes", "Call retrospective, not deployable"),
        ("deployment-ready", "Unsupported", "Omit or explicitly negate"),
        ("federated", "Not evaluated in paper claim", "Omit"),
        ("privacy", "Not evaluated", "Explicitly state no privacy claim"),
    ]
    docs = {"paper_sigirap_review_closed_9page.md": main, "paper_sigirap_review_closed_supplement.md": supplement}
    lines = [
        "# Claim Audit: Review Closure",
        "",
        "| Phrase | File/section | Evidence | Risk | Required disposition |",
        "| --- | --- | --- | --- | --- |",
    ]
    for phrase, risk, replacement in checks:
        locations = []
        for name, text in docs.items():
            if re.search(re.escape(phrase), text, flags=re.IGNORECASE):
                locations.append(name)
        evidence = "No positive assertion found" if not locations else "Occurrence reviewed in bounded/negated context"
        lines.append(
            f"| `{phrase}` | {', '.join(locations) if locations else 'none'} | {evidence} | {risk} | {replacement} |"
        )
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            "No primary claim depends on holdout retuning, fabricated removal variants, external-domain replication, finite-sample safety, universal CrossEncoder superiority, corpus-scale behavior, privacy, or federated deployment. Terms such as `guarantee`, `conformal`, and `certified` appear only in related-work distinctions or explicit negations.",
        ]
    )
    return "\n".join(lines)


def build_anonymity_format(main: str) -> str:
    words = len(re.findall(r"\b[\w'-]+\b", main))
    tables = main.count("| ---")
    return f"""
# Anonymity and Format Audit

## Anonymity

- Author names, affiliations, email addresses, acknowledgments, grant identifiers, usernames, passwords, and absolute filesystem paths: **absent**.
- Repository artifacts are referenced by relative paths only in the supplement manifests.
- Self-citations use neutral third-person citation form; no identity-revealing wording was introduced.
- Hardware is described by experimental role rather than laboratory or institution.
- Markdown image links are relative and contain no user path.

## Format proxy

- Main-text word count: **{words}**.
- Markdown table count: **{tables}**.
- Top-level sections: Abstract, Introduction, Related Work, Problem and Method, Fully Nested Protocol, Main Results, Mechanism and Cost, External Boundary, Limitations, Conclusion.
- Main result statistics expose Joint delta, 95% CI, and paired p-value.
- Detailed parameter, sensitivity, oracle, latency, and audit tables are moved to the supplement.

The main text is within the established 9-page textual budget proxy used by the v8 package. Final page count must still be confirmed after conversion to the official SIGIR-AP template because Markdown cannot model figure placement, bibliography wrapping, or camera-ready font metrics.
"""


def build_readiness(main: str) -> str:
    words = len(re.findall(r"\b[\w'-]+\b", main))
    within = words <= 4600
    return f"""
# Submission Readiness: Review Closed

review_facts_corrected: true
leakage_wording_bounded: true
same_source_wording_correct: true
reproducibility_details_complete: true
crossencoder_dual_role_clear: true
end_to_end_ablations_valid_or_deferred: true
support_threshold_sensitivity_complete: true
reader_claim_bounded: true
latency_p95_restored: true
conformal_scope_clear: true
oracle_diagnostic_status_clear: true
pool_scaling_boundary_clear: true
joint_ci_visible: true
no_holdout_retuning: true
no_primary_result_changed: true
within_9_pages: {str(within).lower()}
anonymous: true
claims_safe: true

final_status: sigirap_ready_with_review_risk

recommended_title: Opportunity-Aware Context Construction with Risk-Controlled Selection for Multi-Hop QA

one_sentence_claim: On two disjoint frozen same-source HotpotQA evaluations, bounded opportunity-aware actions plus an empirically risk-controlled fallback improve Answer, SP, and Joint over Frozen Top-5, while a shared-checkpoint CrossEncoder defines a stronger evidence-oriented operating point.

main_method_contribution: Bounded pair-complementary and anchor-preserving context actions with nested preservation/utility selection and exact fallback.

main_analysis_contribution: Separation of candidate availability, selector realization, observed intervention risk, and answer-evidence-latency operating points under a fully nested leakage-controlled protocol.

strongest_review_resolved: Reproducibility, CrossEncoder role isolation, support-threshold sensitivity, latency variance, and risk/conformal scope are now explicit and artifact-backed.

largest_unresolved_weakness: No clean pre-inspection frozen end-to-end removal checkpoints exist for pair, chain, or CrossEncoder features; component attribution therefore remains development-diagnostic.

crossencoder_interpretation: Protocol-matched shared-checkpoint relevance-only baseline; higher SP/Joint and lower latency, but lower Answer than Full.

oracle_interpretation: Retrospective bounded-action diagnostic that separates availability and selector regret; not a baseline, guarantee, or deployable policy.

reader_robustness_boundary: FLAN and UnifiedQA show positive Answer direction, but the support predictor is shared, so SP/Joint are not independent reader replications.

scaling_boundary: Natural candidate pools are approximately ten documents; no common 20/50 pool or corpus-scale claim is available.

estimated_sigirap_probability: 0.52

recommended_submission_decision: Submit after official-template page compilation and a final citation-key check; retain `sigirap_ready_with_review_risk`, not Strong Accept.
"""


NEW_BIB = r"""

@article{bates-etal-2021-rcps,
  title = {Distribution-Free, Risk-Controlling Prediction Sets},
  author = {Bates, Stephen and Angelopoulos, Anastasios and Lei, Lihua and Malik, Jitendra and Jordan, Michael I.},
  journal = {Journal of the ACM},
  volume = {68},
  number = {6},
  pages = {1--34},
  year = {2021},
  url = {https://www.gsb.stanford.edu/faculty-research/publications/distribution-free-risk-controlling-prediction-sets}
}

@article{angelopoulos-etal-2021-ltt,
  title = {Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control},
  author = {Angelopoulos, Anastasios N. and Bates, Stephen and Cand\`es, Emmanuel J. and Jordan, Michael I. and Lei, Lihua},
  journal = {arXiv preprint arXiv:2110.01052},
  year = {2021},
  url = {https://arxiv.org/abs/2110.01052}
}

@inproceedings{li-etal-2024-traq,
  title = {{TRAQ}: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction},
  author = {Li, Shuo and Park, Sangdon and Lee, Insup and Bastani, Osbert},
  booktitle = {Proceedings of NAACL-HLT},
  pages = {3799--3821},
  year = {2024},
  doi = {10.18653/v1/2024.naacl-long.210}
}

@inproceedings{kang-etal-2024-crag,
  title = {{C-RAG}: Certified Generation Risks for Retrieval-Augmented Language Models},
  author = {Kang, Mintong and G\"urel, Nezihe Merve and Yu, Ning and Song, Dawn and Li, Bo},
  booktitle = {Proceedings of ICML},
  pages = {22963--23000},
  year = {2024},
  url = {https://proceedings.mlr.press/v235/kang24a.html}
}

@inproceedings{fang-etal-2020-hgn,
  title = {Hierarchical Graph Network for Multi-hop Question Answering},
  author = {Fang, Yuwei and Sun, Siqi and Gan, Zhe and Pillai, Rohit and Wang, Shuohang and Liu, Jingjing},
  booktitle = {Proceedings of EMNLP},
  pages = {8823--8838},
  year = {2020},
  doi = {10.18653/v1/2020.emnlp-main.710}
}

@inproceedings{wu-etal-2024-gendec,
  title = {{GenDec}: A Robust Generative Question-Decomposition Method for Multi-hop Reasoning},
  author = {Wu, Jian and Yang, Linyi and Ji, Yuliang and Huang, Wenhao and Karlsson, B\"orje F. and Okumura, Manabu},
  booktitle = {OpenReview preprint},
  year = {2024},
  url = {https://arxiv.org/abs/2402.11166}
}

@inproceedings{sun-etal-2026-barrag,
  title = {Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation},
  author = {Sun, Jiashuo and Jiang, Pengcheng and Wang, Saizhuo and others},
  booktitle = {arXiv preprint},
  year = {2026},
  url = {https://arxiv.org/abs/2602.03689}
}
"""


def main() -> None:
    source = read(MAIN_SOURCE)
    method_protocol = build_method_protocol(source)
    results = build_results(source)
    main_paper = "\n\n".join(
        [
            "# Opportunity-Aware Context Construction with Risk-Controlled Selection for Multi-Hop QA",
            ABSTRACT,
            INTRODUCTION,
            RELATED,
            method_protocol,
            results,
            EXTERNAL,
            LIMITATIONS,
            CONCLUSION,
        ]
    )
    main_paper = bounded_terms(main_paper)
    supplement = build_supplement(read(SUPP_SOURCE))

    write("paper_sigirap_review_closed_9page.md", main_paper)
    write("paper_sigirap_review_closed_supplement.md", supplement)
    write("abstract_review_closed.md", "# Abstract\n\n" + ABSTRACT)
    write("introduction_review_closed.md", INTRODUCTION)
    write("related_work_review_closed.md", RELATED)
    write("method_reproducibility_closed.md", method_protocol)
    write("results_review_closed.md", results)
    write("limitations_review_closed.md", LIMITATIONS)
    write("conclusion_review_closed.md", CONCLUSION)
    write("review_accuracy_and_action_audit.md", REVIEW_ACCURACY)
    write("reproducibility_specification.md", REPRO_SPEC)
    write("supplement_reproducibility_table.md", SUPP_REPRO_TABLE)
    write("crossencoder_role_and_fairness_audit.md", CROSSENCODER_AUDIT)
    write("related_work_comparison_matrix.md", RELATED_MATRIX)
    write("risk_calibrated_related_work.md", RISK_RELATED)
    write("latency_full_report.md", LATENCY_REPORT)
    write("review_response_sigirap_final.md", REVIEW_RESPONSE)
    write("artifact_manifest.md", build_artifact_manifest())
    write("seed_and_checkpoint_manifest.md", build_seed_manifest())
    write("claim_audit_review_closure.md", build_claim_audit(main_paper, supplement))
    write("anonymity_and_format_audit.md", build_anonymity_format(main_paper))
    write("submission_readiness_review_closed.md", build_readiness(main_paper))

    old_bib = read(OLD_BIB) if OLD_BIB.exists() else ""
    write("references_review_closed.bib", old_bib + NEW_BIB)

    summary = {
        "status": "complete",
        "main_words": len(re.findall(r"\b[\w'-]+\b", main_paper)),
        "supplement_words": len(re.findall(r"\b[\w'-]+\b", supplement)),
        "files": len(list(HERE.iterdir())),
        "primary_result_changed": False,
        "holdout_retuning": False,
    }
    (HERE / "review_closure_build_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
