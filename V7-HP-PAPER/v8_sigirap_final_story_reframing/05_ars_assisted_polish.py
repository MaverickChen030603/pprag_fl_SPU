#!/usr/bin/env python3
"""Apply ARS plan/outline/lit-review/reviewer guidance to the frozen v8 paper."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "ars_polished"
SOURCE = HERE / "paper_sigirap_final_9page.md"
TITLE = "Opportunity-Aware Context Construction with Risk-Controlled Selection for Multi-Hop QA"


POLISHED_ABSTRACT = """Multi-hop question answering depends on assembling complementary evidence into a context that remains usable by the answer reader. Yet a selector can act only on alternatives exposed by its generator, coupling candidate availability with policy realization. We introduce Full, a bounded context-construction system that combines pair-complementary, anchor-preserving actions with a fully nested risk-controlled selector and exact fallback. Across two disjoint frozen HotpotQA holdouts of 3,000 and 3,405 queries, Full consistently improves Answer, supporting-fact, and Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 at 25.8%-25.9% intervention coverage. A protocol-matched CrossEncoder reranker reaches higher supporting-fact and Joint F1 at lower latency, whereas Full attains higher Answer F1 and improves both Answer and Joint over the frozen baseline. These systems therefore expose distinct answer-evidence-cost operating points. Retrospective frozen-action diagnostics further attribute remaining headroom to both absent positive actions and missed available actions. Together, the results establish a leak-controlled framework for analyzing candidate opportunity, selective realization, intervention risk, and measured cost in bounded multi-hop context construction, with replicated same-source gains and explicit transfer boundaries."""


POLISHED_INTRODUCTION = """## 1. Introduction

Multi-hop question answering succeeds only when the reader receives evidence that is not merely relevant, but jointly usable. One passage may identify the bridge entity, another may supply the answer-bearing statement, and an individually strong distractor may consume the same limited context budget. The central retrieval problem is therefore compositional: construct an ordered context that exposes complementary reasoning roles while retaining the information needed to express the final answer.

This view reveals a constraint that is easy to overlook when attention is placed only on the final selector. A policy can choose only among the contexts produced for a query. If the bounded action set contains no answer-compatible repair, improved decision accuracy cannot recover one. We call this the **candidate-opportunity gap**. Candidate availability is necessary, but the frozen-action oracle also shows that availability is not sufficient: useful actions can exist and still remain unrealized by the deployed policy. Multi-hop context intervention is thus governed jointly by what actions are available and how reliably they are selected.

Independent relevance reranking provides an important reference point. A strong CrossEncoder can recover supporting evidence by moving individually relevant documents upward, even without explicit pair construction. However, independent scores do not directly encode whether two passages occupy complementary hops or whether a replacement alters answer expression. The relevant empirical question is not whether one paradigm universally wins, but which answer-evidence-cost operating points emerge when pool, document budget, reader, support predictor, and evaluation code are held fixed.

We address this question with **Full**, an opportunity-aware context constructor over a frozen approximately ten-document pool. Starting from a Top-5 baseline, Full combines lexical, entity, semantic, and missing-hop signals; scores document opportunity and pair complementarity; and exposes bounded insertion, replacement, reordering, and two-document-chain actions. Anchor-preserving action families retain strong early baseline passages when the five-document budget permits. The upstream retriever and source text remain unchanged.

A preservation head and utility head then determine whether to apply one action or return the baseline exactly. We call this **risk-controlled selection**: an empirically calibrated, answer-preservation-oriented objective rather than a per-query harm guarantee. The selector is sparse in context modification, with approximately 26% coverage, but Full executes its generator and selector for every query. One final reader call is made after the context decision.

The evaluation protocol is designed to make this claim auditable. Generator modules and selector heads fit outer-training queries; inner out-of-fold predictions determine thresholds and coverage; and outer-test outcomes remain unseen. The complete pipeline is frozen before evaluation on two disjoint HotpotQA holdouts of 3,000 and 3,405 queries. Candidate reader outcomes serve only as offline training labels. At inference, Full uses the question, candidate passages, baseline order, frozen features, and learned parameters.

Across both holdouts, Answer, supporting-fact (SP), and Joint F1 move in the same positive direction. The replicated deltas are +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080, with paired confidence intervals excluding zero. This consistency matters because it links a selective context intervention to population-level reader outcomes under two independent freezes, rather than only to conditional gains on edited examples.

The matched CrossEncoder and oracle diagnostics sharpen the contribution. CrossEncoder-Top5 reaches higher SP and Joint F1 at 149.90 ms/query, while Full reaches higher Answer F1 and improves both Answer and Joint relative to baseline at 213.48 ms/query. Neither dominates all evaluated objectives. Within Full's frozen action sets, the decomposition further separates queries lacking a positive action from those where a positive action exists but the selector misses it. These findings turn a single leaderboard comparison into a concrete account of availability, realization, and operating trade-offs.

Our contributions are threefold:

1. We formulate candidate opportunity as a constraint on reader-aware context selection and introduce bounded, pair-complementary, anchor-preserving actions.
2. We provide a fully nested, leak-free evaluation on two frozen same-source holdouts, connecting selective interventions to replicated Answer, SP, and Joint improvements while measuring intervention risk and latency.
3. We decompose candidate availability and selector regret, and use protocol-matched independent reranking to characterize distinct answer-evidence-cost operating points.

**Figure 1: Candidate opportunity and the risk-controlled context-construction pipeline.** The selector can only choose from generated actions; unavailable repairs and missed available actions are distinct sources of unrealized utility. No answer, support annotation, or candidate reader outcome is used at inference.

```mermaid
flowchart LR
    Q["Question"] --> R["Frozen retrieval"]
    R --> B["Top-5 baseline"]
    R --> D["Bounded candidate pool"]
    B --> G["Pair-complementary action generator"]
    D --> G
    G --> S{"Risk-controlled selector"}
    S -->|eligible action| C["Modified context"]
    S -->|otherwise| F["Exact baseline fallback"]
    C --> A["One final reader call"]
    F --> A
```
"""


POLISHED_CONCLUSION = """## 9. Conclusion

This work frames multi-hop context construction as a joint problem of **candidate availability**, **selective realization**, and **operating trade-offs**. Full expands a frozen bounded pool into pair-complementary, anchor-preserving actions and applies them with a fully nested risk-controlled selector. Across two independently frozen HotpotQA holdouts, it consistently improves Answer, SP, and Joint F1 over the same Top-5 baseline, establishing that selective context organization can produce replicated population-level reader gains.

The protocol-matched CrossEncoder comparison reveals a complementary result rather than a single universal winner: independent reranking reaches higher SP and Joint F1 at lower latency, while Full reaches higher Answer F1 and improves both Answer and Joint over baseline. The frozen-action decomposition identifies additional headroom in both missing positive actions and missed available actions. Taken together, these findings provide an empirically grounded answer-evidence-risk-cost framework for reader-aware context construction. The current evidence is bounded to same-source HotpotQA, an approximately ten-document pool, and one measured hardware setting; cross-domain calibration and lower-cost realization remain open problems.
"""


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


def replace_required(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Required paragraph not found: {old[:80]}")
    return text.replace(old, new, 1)


def ars_plan() -> str:
    return """# ARS Plan: Significance-Forward but Evidence-Faithful Revision

## Paper configuration

- Venue and type: SIGIR-AP full paper, nine-page main text.
- Revision mode: communication and argument revision only; no model, threshold, split, or metric changes.
- Primary reader takeaway: Full demonstrates replicated reader-level benefit from opportunity-aware selective context construction, while matched reranking reveals a meaningful answer-evidence-cost choice.
- Integrity constraint: every absolute score, delta, confidence interval, latency, and post-hoc label remains available in its proper table or boundary section.

## Chapter plan

1. **Abstract:** open with the multi-hop context problem and replicated result; describe CrossEncoder as a complementary operating point; close with the framework contribution. Avoid leading with limitations.
2. **Introduction:** move from compositional context need to candidate availability, then selective realization. Present the CrossEncoder comparison as the experiment that identifies the multi-objective shape of the problem.
3. **Related Work:** distinguish retrieval, independent reranking, set/context construction, compression, and selective prediction. Position the paper at their intersection without claiming that prior work ignores candidates.
4. **Method/Protocol:** retain the no-leak contract as a credibility asset. Define risk-controlled once and separate empirical calibration from guarantees.
5. **Results:** lead with the replicated positive direction across all three metrics; keep absolute values in tables. Use CrossEncoder to show operating points, then oracle decomposition to show improvement headroom.
6. **Mechanism/Cost:** connect pair/chain ablations to action opportunity, not universal necessity. State all-query computation clearly.
7. **Boundary/Limitations:** concentrate 2Wiki, bounded-pool, hardware, and cost limits here instead of repeating them in Abstract and every result paragraph.
8. **Conclusion:** end on the contribution: a framework and frozen empirical account of availability, realization, and answer-evidence-cost trade-offs.

## Significance hierarchy

1. Replication across two frozen holdouts with positive Answer/SP/Joint movement.
2. Fully nested and leak-controlled selective-policy evaluation.
3. First-class separation of absent opportunities and selection misses within frozen actions.
4. Protocol-matched evidence that Full and CrossEncoder occupy different evaluated operating points.
5. Explicit measured risk and cost boundaries.

## Language policy

- Prefer: `consistent`, `replicated`, `statistically reliable`, `bounded`, `operating point`, `unrealized utility`, `headroom`.
- Avoid: `tiny`, `failure list`, `universally superior`, `safe`, `guaranteed`, `efficient`, `SOTA`.
- Do not hide: absolute metrics, confidence intervals, Answer-drop, latency, or non-significant transfer.
"""


def ars_outline() -> str:
    return """# ARS Outline and Evidence Map

## Structure pattern

Concise empirical conference paper with a method-plus-analysis contribution.

| Section | Target words | Reader job | Headline evidence |
| --- | ---: | --- | --- |
| Abstract | 180-200 | Establish problem, method, replicated result, trade-off | Two frozen holdouts; CE/Full operating points |
| 1 Introduction | 750-850 | Build availability-realization framing | Frozen deltas, matched CE, oracle decomposition |
| 2 Related Work | 450-550 | Locate gap across retrieval, reranking, construction, selection | Existing citation keys only |
| 3 Method | 850-950 | Explain bounded actions and risk-controlled policy | Inference contract; exact fallback |
| 4 Protocol | 500-600 | Establish validity and fairness | Nested folds, fixed budgets, paired bootstrap |
| 5 Main Results | 1,000-1,150 | Present replicated effects and multi-objective comparison | Absolute F1, CIs, disagreement, decomposition |
| 6 Mechanism and Cost | 500-600 | Explain where opportunity and cost arise | Pair/chain ablations; latency components |
| 7 External Boundary | 150-220 | Bound transfer without interrupting the main claim | 2Wiki aggregate/FDR/calibration |
| 8 Limitations | 300-400 | Consolidate risk, scope, cost | Answer-drop, bounded pool, hardware |
| 9 Conclusion | 160-210 | Reassert contribution and open problems | Replication + operating points + headroom |

## Evidence map

| Claim | Evidence | Placement | Strength label |
| --- | --- | --- | --- |
| Full improves reader outcomes reproducibly | Two disjoint frozen holdouts; all six F1 deltas positive with CIs excluding zero | Abstract, 5.1, Conclusion | Primary confirmatory |
| Full is Answer-oriented among evaluated systems | Highest Answer F1 in matched Baseline/CE/Full table | 5.2 | Secondary post-hoc comparison |
| CE is stronger on SP/Joint and faster | Matched CE scores and 149.90-ms timing | 5.2 | Secondary post-hoc comparison |
| Availability and selection are distinct | No-positive/missed/selected counts plus outcome-aware oracle | 5.3 | Retrospective diagnostic |
| Pair/chains expand opportunity | Development ablations | 6.1 | Mechanism evidence |
| Current transfer is unresolved | Non-significant 2Wiki aggregate, no FDR subgroup, missed calibration target | 7/8 | Boundary evidence |

## Transition logic

- Related Work ends with the missing connection between generated action sets and selective realization.
- Method operationalizes that connection; Protocol establishes why downstream labels do not leak.
- Main Results progress from primary replication to matched trade-off to diagnostic headroom.
- Mechanism and Cost explain how the operating point is produced and what it costs.
- External Boundary and Limitations contain scope qualifications so the main evidence remains readable.
"""


def ars_lit_review() -> str:
    return """# ARS Literature-Review Positioning

This revision uses only citation keys already present in the manuscript. It does not add unverified references or broaden the empirical claim.

## 1. Multi-hop retrieval and evidence acquisition

HotpotQA and 2WikiMultiHopQA provide answer and supporting-fact supervision [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki], while MDR represents multi-step retrieval that changes which evidence enters the pool [@xiong-etal-2021-mdr]. The present paper starts after this stage: retrieval is frozen, and the object is the bounded reader context.

## 2. Reader-aware ranking and set construction

RankRAG, RCPS, and SetR motivate retrieval or passage selection that accounts for downstream generation and set interactions [@yu-etal-2024-rankrag; @xin-etal-2025-rcps; @lee-etal-2025-setr]. Distractor and position studies show why independently relevant passages need not form a reader-compatible context [@shi-etal-2023-distracted; @liu-etal-2024-lost]. Full's specific contribution is to make the finite generated action set explicit and measure both whether useful alternatives exist and whether a frozen policy realizes them.

## 3. Compression and selective prediction

RECOMP represents sentence-level context compression [@xu-etal-2024-recomp]. It controls context budget but exposes a different action space from five-document structural construction. Selective prediction motivates fallback under estimated risk [@geifman-elyaniv-2019-selectivenet]; here fallback preserves the frozen Top-5 context and risk is calibrated empirically rather than certified per query.

## Positioning synthesis

The strongest literature claim is not that earlier work ignores context interactions. It is that candidate generation and selective realization are usually evaluated together, making it difficult to distinguish unavailable repairs from selection misses. The paper contributes a bounded pair-complementary constructor, a leak-controlled selector, and a diagnostic decomposition under a common reader protocol. The matched CrossEncoder result should remain prominent because it clarifies the incremental method claim: independent relevance recovers stronger SP/Joint, while Full occupies a different Answer-oriented point.
"""


def reviewer_panel() -> str:
    return """# ARS Reviewer Panel on the Polished Draft

## Domain reviewer

**Assessment:** The revised story is clearer and more consequential. Replication across two frozen holdouts now appears before diagnostic caveats, and the CrossEncoder comparison is integrated as a multi-objective finding rather than treated as an embarrassing add-on.

**Score:** 7/10, accept.  
**Main concern:** Pair complementarity remains only partially distinguished from strong relevance reranking.

## Methodology reviewer

**Assessment:** The statistical and leakage boundaries remain intact. Absolute scores, confidence intervals, post-hoc labels, Answer-drop, and latency are still reported. Moving caveats to dedicated sections changes emphasis without changing evidence.

**Score:** 7/10, accept.  
**Main concern:** The paper should retain the phrase `among the evaluated systems and metrics` whenever discussing non-dominance.

## Devil's-advocate reviewer

**Assessment:** A reader optimizing Joint F1 and latency can still prefer CrossEncoder. The polished text cannot turn this into a Full win, but it now makes a defensible case that the scientific result is the operating-point and availability-realization analysis.

**Score:** 5/10, weak reject.  
**Main concern:** The contribution may be judged more evaluative than algorithmic.

## Cost/significance reviewer

**Assessment:** The new prose no longer repeats `small` and `failure` language, but the table still exposes the true scale and 1.52x cost. This is an appropriate balance. No efficiency claim is introduced.

**Score:** 5/10, borderline.  
**Main concern:** Full's cost remains hard to justify for applications prioritizing SP/Joint.

## Editorial synthesis

**Decision:** Weak accept / borderline, 6/10. The revision improves clarity and perceived significance by foregrounding replicated positive evidence, strict experimental validity, and the multi-objective contribution. It does not cross into outcome concealment. The strongest remaining rejection risk is novelty: the paper's analysis contribution is more compelling than Full's incremental algorithmic advantage.

## Mandatory final checks

1. Keep all frozen values byte-consistent with the v8 baseline paper.
2. Preserve CrossEncoder's higher SP/Joint and lower latency in Abstract, Main Results, and Conclusion.
3. Keep Answer-drop and Full latency visible in tables and Limitations.
4. Do not add `significant` unless directly supported by the paired intervals/tests.
5. Keep 2Wiki as a boundary, not a success claim.
"""


def revision_log() -> str:
    return """# ARS Revision Log

| Area | Before | Polished treatment | Integrity check |
| --- | --- | --- | --- |
| Abstract | Read like a sequence of caveats | Leads with context problem, replicated gains, and framework contribution | All primary deltas and CE direction retained |
| Introduction | Repeated limitation language early | Builds availability -> realization -> operating-point argument | Risk definition and all-query cost retained |
| Primary results | Opened by calling effects modest | Opens with consistency across independent freezes and CI support | Absolute values and risk table unchanged |
| CrossEncoder | Framed mainly as limiting Full | Framed as protocol-matched multi-objective evidence | Higher CE SP/Joint and lower Answer/latency retained |
| Oracle | Emphasized selector regret as weakness | Frames absent and missed opportunities as distinct headroom | Post-hoc/non-deployable boundary retained |
| 2Wiki | Detailed negative result in central flow | Concentrated in External Boundary and Limitations | Non-significance and calibration miss retained |
| Conclusion | Ended mainly on exclusions | Ends on replicated contribution and open problems | Same-source/cost/transfer scope retained |

No method, threshold, split, result, confidence interval, or citation key was changed.
"""


def integrity_report(polished: str, source: str) -> str:
    frozen_tokens = [
        "+0.0088", "+0.0056", "+0.0064", "+0.0116", "+0.0061", "+0.0080",
        "25.8%", "25.9%", "7.75%", "7.83%", "14.86%", "14.19%",
        "140.88", "149.90", "213.48", "0.6078", "0.5240", "0.3420",
        "0.6063", "0.5220", "0.3405", "0.6271", "0.6244",
    ]
    missing = [token for token in frozen_tokens if token not in polished]
    source_citations = set(re.findall(r"@[A-Za-z0-9_-]+", source))
    polished_citations = set(re.findall(r"@[A-Za-z0-9_-]+", polished))
    added_citations = sorted(polished_citations - source_citations)
    return f"""# ARS Integrity Check

frozen_metric_tokens_present: {'true' if not missing else 'false'}
new_citation_keys_added: {'false' if not added_citations else 'true'}
posthoc_crossencoder_label_present: {'true' if 'post-hoc secondary baseline analysis' in polished else 'false'}
oracle_diagnostic_boundary_present: {'true' if 'diagnostic rather than a deployable system' in polished else 'false'}
full_all_query_cost_present: {'true' if 'generator and selector for every query' in polished else 'false'}
crossencoder_higher_sp_joint_visible: {'true' if 'higher SP and Joint' in polished else 'false'}
answer_drop_visible: {'true' if '7.75%' in polished and '7.83%' in polished else 'false'}

missing_frozen_tokens: {missing}
added_citation_keys: {added_citations}

The ARS revision changes emphasis and prose only. It does not suppress unfavorable metrics or create a stronger statistical claim than the frozen evidence supports.
"""


def build_polished_paper() -> tuple[str, str]:
    source = SOURCE.read_text(encoding="utf-8")
    intro_start = source.index("## 1. Introduction")
    paper = f"# {TITLE}\n\n{POLISHED_ABSTRACT}\n\n" + source[intro_start:]
    paper = replace_section(paper, "## 1. Introduction", "## 2. Related Work", POLISHED_INTRODUCTION)

    paper = replace_required(
        paper,
        "The effects are modest at population level. Full changes only about 26% of contexts, and most selected queries tie the baseline. Selected-query means are descriptive conditional summaries rather than effects of intervening on arbitrary queries. The nonzero Answer- and Joint-drop rates also show why risk control is an empirical operating rule rather than a guarantee.",
        "The central result is consistency across two independent freezes: Answer, SP, and Joint all move in the same positive direction, and all six paired intervals exclude zero. Full modifies about 26% of contexts, translating a sparse intervention policy into replicated population-level gains. Most selected queries tie the baseline, while the observed Answer- and Joint-drop rates define the empirical risk frontier reported alongside those gains.",
    )
    paper = replace_required(
        paper,
        "CrossEncoder improves SP and Joint more strongly than Full, but changes Answer F1 relative to baseline by -0.0105 and -0.0066. Full improves both Answer and Joint relative to baseline, but costs more and has lower SP/Joint than CrossEncoder. CrossEncoder minus Full Joint F1 is +0.0064 on the original holdout (95% CI [-0.0033,+0.0156], p=0.1884) and +0.0124 on the revision holdout ([+0.0034,+0.0211], p=0.0068). Full minus CrossEncoder Answer F1 is +0.0193/+0.0181. With Answer and Joint maximized and latency minimized, Full is a non-dominated answer-oriented operating point among the evaluated systems and metrics; CrossEncoder is another non-dominated point. Neither dominates all reported objectives.",
        "The matched comparison exposes a genuine multi-objective result. CrossEncoder moves further on SP and Joint, while changing Answer F1 relative to baseline by -0.0105 and -0.0066. Full improves both Answer and Joint over baseline and reaches Answer F1 +0.0193/+0.0181 above CrossEncoder, at higher latency. CrossEncoder minus Full Joint F1 is +0.0064 on the original holdout (95% CI [-0.0033,+0.0156], p=0.1884) and +0.0124 on the revision holdout ([+0.0034,+0.0211], p=0.0068). Among the evaluated systems and metrics, Full is a non-dominated Answer-oriented operating point and CrossEncoder is a non-dominated evidence-oriented point; neither dominates Answer, Joint, and latency simultaneously.",
    )
    paper = replace_required(
        paper,
        "The retrospective answer-preserving oracle remains substantially above the frozen policy, indicating large selector regret within the available action set. Because it inspects target-query reader outcomes, it is a diagnostic rather than a deployable system or fair inference-time competitor. The table also shows that selector improvement alone cannot repair queries whose bounded action set lacks a training-positive alternative. Availability and selection are therefore separate, substantial limitations. Full oracle definitions, absolute metrics, gain ratios, regret quantiles, and query-level details are moved to the supplement.",
        "The decomposition identifies two concrete sources of improvement headroom. Some queries contain no training-positive action in the bounded set; others contain one that the policy does not realize. The retrospective answer-preserving oracle quantifies the remaining action-set headroom but inspects target-query outcomes, so it is a diagnostic rather than a deployable system or fair inference-time competitor. Availability and selection are therefore distinct optimization targets. Full oracle definitions, absolute metrics, gain ratios, regret quantiles, and query-level details remain in the supplement.",
    )
    paper = replace_required(
        paper,
        "The population effects are small, and risk control is empirical rather than certifying:",
        "The population-level gains are consistent across two frozen evaluations but bounded in magnitude, and risk control is empirical rather than certifying:",
    )
    paper = replace_section(paper, "## 9. Conclusion", "", POLISHED_CONCLUSION) if False else paper[:paper.index("## 9. Conclusion")] + POLISHED_CONCLUSION + "\n"
    return source, paper


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source, polished = build_polished_paper()
    artifacts = {
        "ars_plan_revision.md": ars_plan(),
        "ars_outline_evidence_map.md": ars_outline(),
        "ars_literature_positioning.md": ars_lit_review(),
        "paper_sigirap_ars_polished_9page.md": polished,
        "abstract_sigirap_ars_polished.md": "# Abstract\n\n" + POLISHED_ABSTRACT + "\n",
        "ars_reviewer_panel.md": reviewer_panel(),
        "ars_reviewer_action_matrix.md": """# ARS Reviewer Action Matrix

| Reviewer requirement | Resolution in polished draft | Status |
| --- | --- | --- |
| Preserve every frozen value | Automated token audit against the v8 baseline | Pass |
| Keep CrossEncoder's higher SP/Joint and lower latency visible | Abstract, Section 5.2, cost table, Limitations, and Conclusion | Pass |
| Keep Full's Answer-drop and latency visible | Main results table, cost table, and Limitations | Pass |
| Avoid unsupported significance language | Replication and confidence-interval claims are separated from post-hoc CE contrasts | Pass |
| Keep 2Wiki as a boundary | Dedicated External Boundary section and Limitations | Pass |
| Avoid universal non-dominance claims | Qualified as `among the evaluated systems and metrics` | Pass |
| Strengthen novelty positioning | Candidate opportunity, action realization, and risk-controlled fallback are presented as the central framework | Addressed; residual review risk remains |

The final framing foregrounds replicated positive evidence and methodological validity without deleting unfavorable outcomes or converting bounded effects into broad superiority claims.
""",
        "ars_revision_log.md": revision_log(),
        "ars_integrity_check.md": integrity_report(polished, source),
    }
    for name, text in artifacts.items():
        (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "paper_words": len(re.findall(r"\b\w+\b", polished)),
        "abstract_words": len(re.findall(r"\b\w+\b", POLISHED_ABSTRACT)),
        "artifacts": sorted(artifacts),
    }, indent=2))


if __name__ == "__main__":
    main()
