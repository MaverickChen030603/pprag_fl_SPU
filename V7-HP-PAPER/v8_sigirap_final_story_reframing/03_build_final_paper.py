#!/usr/bin/env python3
"""Assemble the final SIGIR-AP story from frozen evidence and v8 analyses."""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "v7_sigirap_targeted_strengthening"
SOURCE_PAPER = SOURCE / "paper_sigirap_strengthened_9page.md"
SOURCE_SUPPLEMENT = SOURCE / "paper_sigirap_strengthened_supplement.md"

TITLE = "Opportunity-Aware Context Construction with Risk-Controlled Selection for Multi-Hop QA"

ABSTRACT = """Multi-hop question answering requires complementary evidence without displacing passages that express the answer. Yet a selector can only choose among contexts exposed by its generator, creating a candidate-opportunity gap. We introduce Full, which constructs bounded pair-complementary, anchor-preserving context actions and applies them through a fully nested risk-controlled policy with exact fallback. Risk-controlled denotes an empirically calibrated, answer-preservation-oriented selection objective; it does not provide a per-query harm guarantee. On two disjoint frozen HotpotQA holdouts of 3,000 and 3,405 queries, Full yields modest Answer/supporting-fact/Joint F1 changes of +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 at approximately 26% intervention coverage. A protocol-matched independent CrossEncoder reranker improves supporting-fact and Joint F1 more strongly but lowers Answer F1, placing Full at a distinct answer-evidence operating point. Retrospective diagnostics show that both missing candidate opportunities and selector regret remain substantial. Full also incurs greater measured post-retrieval latency than the baseline and independent reranker. The evidence establishes a bounded same-source answer-evidence-risk-cost trade-off, not universal reranking superiority, low-cost deployment, or cross-domain reliability."""

INTRODUCTION = """## 1. Introduction

Multi-hop question answering requires more than collecting passages that are individually relevant. A reader must receive complementary evidence together, within a fixed context budget and in a usable order. One passage may establish an entity relation, another may contain the answer-bearing statement, and a third may be a highly ranked distractor. Improving evidence coverage can therefore help supporting-fact prediction while also displacing wording that the answer reader needs. The optimization object is an ordered reader context, not merely a list of relevance scores.

A learned selector faces an upstream constraint: it can choose only among the actions generated for a query. If every proposed insertion, replacement, or reordering omits one hop or removes a useful baseline passage, even a perfect selector over that set cannot recover a compatible context. We call the difference between the available bounded actions and useful reader-compatible alternatives the **candidate-opportunity gap**. An unavailable repair cannot be selected, but availability alone is insufficient because the frozen policy realizes only a limited share of retrospective action-set utility.

Independent relevance reranking offers a strong alternative. A CrossEncoder can move individually relevant documents toward the top and may recover supporting evidence without explicitly constructing pairs. It does not, however, directly represent whether two moderate-scoring passages play complementary hops or whether replacing a baseline passage changes answer expression. This distinction is empirical rather than absolute: a strong independent reranker may still recover much of the downstream gain. The paper therefore evaluates independent relevance and pair-complementary construction under the same pool, document budget, reader, support predictor, and official metrics.

Our Full system starts from a frozen Top-5 context and an approximately ten-document post-retrieval pool. It scores document opportunity and pair complementarity, constructs bounded two-document chains, and retains strong early baseline anchors when the five-document budget permits. The action generator never edits source text or changes the upstream retriever. Its purpose is to expose a small set of structurally different contexts that an independent document ordering may not represent.

A separate preservation head and utility head decide whether one generated action should replace the baseline. If both frozen gates and the fold-level coverage budget pass, the policy applies the highest-ranked eligible action; otherwise it returns the original Top-5 context exactly. We call this **risk-controlled selection**. Risk-controlled denotes an empirically calibrated, answer-preservation-oriented selection objective; it does not provide a per-query harm guarantee. Full is selective in modifying contexts, not in executing its generator and selector, which run for every query.

Evaluation is fully nested to prevent reader outcomes from leaking into test decisions. Generator modules and selector heads fit outer-training queries, inner out-of-fold predictions determine thresholds and coverage, and outer-test outcomes remain unseen. The complete pipeline is then frozen before two disjoint same-source HotpotQA evaluations of 3,000 and 3,405 queries. Candidate reader outcomes are offline labels only; inference uses question, passages, baseline order, frozen features, and learned parameters, followed by one final reader call.

Full produces modest replicated population gains. Answer/supporting-fact (SP)/Joint F1 change by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 at 25.8%-25.9% intervention coverage. Selected Answer F1 decreases on 7.75%-7.83% of interventions, and measured post-retrieval latency rises from 140.88 to 213.48 ms/query. These are same-source quality-risk-cost results, not broad safety or efficiency claims.

Two post-hoc analyses change how the result should be interpreted. First, protocol-matched CrossEncoder-Top5 reaches higher SP and Joint F1 at 149.90 ms/query, but its Answer F1 is below both Full and the frozen baseline. Full and CrossEncoder thus occupy different answer-evidence-latency operating points; neither dominates across all reported objectives. Second, an outcome-aware oracle restricted to the existing action set remains substantially above the frozen policy. The new diagnostics show that neither action construction nor policy selection alone determines performance: candidate availability and selector regret are separate bottlenecks.

Our contributions are threefold:

1. We formulate candidate opportunity as a constraint on reader-aware context selection and introduce bounded, pair-complementary, anchor-preserving actions.
2. We provide a fully nested, leak-free selective-policy evaluation on two frozen same-source holdouts, separating population effects, intervention risk, and measured cost.
3. We decompose candidate availability and selector regret, and compare Full with protocol-matched independent relevance reranking to characterize an answer-evidence-cost trade-off.

**Figure 1: Candidate opportunity and the risk-controlled context-construction pipeline.** The selector can only choose from generated actions; unavailable repairs and missed available actions are distinct failure modes. No answer, support annotation, or candidate reader outcome is used at inference.

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

CONCLUSION = """## 9. Conclusion

Multi-hop context intervention is limited jointly by the actions made available and the ability of a frozen selector to identify useful, answer-compatible choices. Full expands the bounded action set with pair-complementary, anchor-preserving contexts and improves Answer and Joint F1 over the frozen baseline on two same-source holdouts. A protocol-matched CrossEncoder reranker instead attains higher SP and Joint F1 at lower latency than Full, while lowering Answer F1 below baseline. Full is therefore a distinct answer-oriented operating point among the evaluated systems and metrics, not a universally superior reranker.

The retrospective oracle further shows substantial selector regret within the available action set, while the decomposition also identifies many queries with no training-positive action. Full incurs 1.52x baseline post-retrieval latency, pair-score pruning saves little, and frozen 2Wiki transfer remains non-significant. The contribution is a fully nested method and analysis of candidate availability, selective realization, and measured answer-evidence-risk-cost trade-offs under a bounded same-source protocol. It does not establish a per-query harm guarantee, corpus-scale efficiency, or cross-domain reliability.
"""


def section(text: str, start: str, end: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[left:right].rstrip()


def disagreement_values() -> dict[str, object]:
    return json.loads((HERE / "outputs/disagreement_metrics.json").read_text(encoding="utf-8"))


def main_results(disagreement: dict[str, object]) -> str:
    events = {}
    for split in ("holdout3000", "revision3405"):
        events[split] = disagreement["splits"][split]["cross_events"]
    return f"""## 5. Main Results

### 5.1 Frozen Same-Source Holdouts

| Frozen split | N | Coverage | Answer delta | SP delta | Joint delta | Selected Answer-drop | Selected Joint-drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original holdout | 3,000 | 25.8% | +0.0088 | +0.0056 | +0.0064 | 7.75% | 14.86% |
| Revision holdout | 3,405 | 25.9% | +0.0116 | +0.0061 | +0.0080 | 7.83% | 14.19% |

Full improves Answer, SP, and Joint F1 on both frozen holdouts. On the original holdout, baseline/Full values are 0.6183/0.6271 for Answer, 0.4930/0.4987 for SP, and 0.3292/0.3356 for Joint. On the revision holdout they are 0.6129/0.6244, 0.4862/0.4923, and 0.3201/0.3280. Paired 95% intervals exclude zero for all six population deltas. Because both samples come from HotpotQA distractor validation, they provide same-source replication rather than external generalization.

The effects are modest at population level. Full changes only about 26% of contexts, and most selected queries tie the baseline. Selected-query means are descriptive conditional summaries rather than effects of intervening on arbitrary queries. The nonzero Answer- and Joint-drop rates also show why risk control is an empirical operating rule rather than a guarantee.

### 5.2 Answer-Evidence Trade-off against Independent Reranking

CrossEncoder-Top5 independently scores every document in the same approximately ten-document pool and retains five. It shares the frozen relevance checkpoint, reader, prompt, 3,200-character cap, support predictor, and official metrics with Full, but excludes pair, missing-hop, outcome-model, and selector features. Score order is chosen on development only and then frozen. Because this comparison was added after the primary study, it is a post-hoc secondary baseline analysis.

| Split | System | Answer F1 | SP F1 | Joint F1 | Latency (ms/query) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original 3,000 | Frozen Top-5 | 0.6183 | 0.4930 | 0.3292 | 140.88 |
| Original 3,000 | CrossEncoder-Top5 | 0.6078 | 0.5240 | 0.3420 | 149.90 |
| Original 3,000 | Full | 0.6271 | 0.4987 | 0.3356 | 213.48 |
| Revision 3,405 | Frozen Top-5 | 0.6129 | 0.4862 | 0.3201 | 140.88 |
| Revision 3,405 | CrossEncoder-Top5 | 0.6063 | 0.5220 | 0.3405 | 149.90 |
| Revision 3,405 | Full | 0.6244 | 0.4923 | 0.3280 | 213.48 |

CrossEncoder improves SP and Joint more strongly than Full, but changes Answer F1 relative to baseline by -0.0105 and -0.0066. Full improves both Answer and Joint relative to baseline, but costs more and has lower SP/Joint than CrossEncoder. CrossEncoder minus Full Joint F1 is +0.0064 on the original holdout (95% CI [-0.0033,+0.0156], p=0.1884) and +0.0124 on the revision holdout ([+0.0034,+0.0211], p=0.0068). Full minus CrossEncoder Answer F1 is +0.0193/+0.0181. With Answer and Joint maximized and latency minimized, Full is a non-dominated answer-oriented operating point among the evaluated systems and metrics; CrossEncoder is another non-dominated point. Neither dominates all reported objectives.

![Answer-Joint-latency operating points](answer_joint_latency_tradeoff.pdf)

**Figure 2:** Frozen operating points. RECOMP-660 appears only on the original holdout and Lite only on the revision holdout because no corresponding frozen result exists on the other split. Non-dominance is assessed only among the evaluated systems, metrics, and measured post-retrieval latency.

The paired outcomes make the aggregate trade-off more precise. CE improves SP while lowering Answer on {events['holdout3000']['ce_sp_up_answer_down']['n']} ({events['holdout3000']['ce_sp_up_answer_down']['proportion']:.1%}) and {events['revision3405']['ce_sp_up_answer_down']['n']} ({events['revision3405']['ce_sp_up_answer_down']['proportion']:.1%}) queries. Direct opposition where Full improves Answer and CE lowers it occurs on only {events['holdout3000']['full_answer_up_ce_answer_down']['n']}/{events['revision3405']['full_answer_up_ce_answer_down']['n']} queries, so the population difference should not be reduced to one common per-query failure mode.

| Post-hoc event | Original 3,000 | Revision 3,405 |
| --- | ---: | ---: |
| CE SP up, Answer down | {events['holdout3000']['ce_sp_up_answer_down']['n']} ({events['holdout3000']['ce_sp_up_answer_down']['proportion']:.1%}) | {events['revision3405']['ce_sp_up_answer_down']['n']} ({events['revision3405']['ce_sp_up_answer_down']['proportion']:.1%}) |
| Full Answer up, CE Answer down | {events['holdout3000']['full_answer_up_ce_answer_down']['n']} ({events['holdout3000']['full_answer_up_ce_answer_down']['proportion']:.1%}) | {events['revision3405']['full_answer_up_ce_answer_down']['n']} ({events['revision3405']['full_answer_up_ce_answer_down']['proportion']:.1%}) |
| Both Joint up | {events['holdout3000']['both_joint_up']['n']} ({events['holdout3000']['both_joint_up']['proportion']:.1%}) | {events['revision3405']['both_joint_up']['n']} ({events['revision3405']['both_joint_up']['proportion']:.1%}) |

This disagreement analysis uses frozen per-query outcomes and is descriptive. The artifacts do not contain a reliable explicit answer-anchor label, so we do not create an outcome-derived proxy or claim a causal anchor mechanism.

### 5.3 Candidate Opportunity and Selector Regret

The frozen action-set decomposition separates queries with no positive action under the training definition from queries where such an action exists but the policy misses it. A positive action is one labeled answer-compatible and utility-improving in the original offline training protocol.

| Split | No positive action | Positive action missed | Positive action selected |
| --- | ---: | ---: | ---: |
| Original 3,000 | 2,316 | 465 | 219 |
| Revision 3,405 | 2,638 | 515 | 252 |

The retrospective answer-preserving oracle remains substantially above the frozen policy, indicating large selector regret within the available action set. Because it inspects target-query reader outcomes, it is a diagnostic rather than a deployable system or fair inference-time competitor. The table also shows that selector improvement alone cannot repair queries whose bounded action set lacks a training-positive alternative. Availability and selection are therefore separate, substantial limitations. Full oracle definitions, absolute metrics, gain ratios, regret quantiles, and query-level details are moved to the supplement.
"""


def mechanism_and_cost() -> str:
    return """## 6. Mechanism and Cost

### 6.1 Core Components

| Generator variant | Positive-action density | Opportunity coverage | Training-label preservation rate | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Full | 14.71% | 29.2% | 92.66% | Frozen joint recipe |
| Without pair complementarity | 10.27% | 27.7% | 93.07% | Clearest learned opportunity loss |
| Without two-document chains | 10.40% | 25.1% | 93.69% | Clearest structural coverage loss |
| Without anchor-preserving families | 16.57% | 27.4% | 92.45% | Higher density but narrower coverage |
| Lite-Lexical-Pair | -- | -- | -- | 0.3217 Joint vs Full 0.3280; NI failed |

Removing pair complementarity or two-document chains produces the clearest development opportunity losses. Removing anchor-preserving families changes both the action denominator and coverage, so its higher positive density is not a monotonic improvement. These outcomes support the frozen joint recipe and pair/chain mechanisms, not the necessity of every semantic feature. Opportunity and preservation rates use offline development outcomes for mechanism analysis; they are not inference-time labels or guarantees.

### 6.2 Quality-Risk-Cost Analysis

| System / boundary | Frozen split | Joint contrast | Coverage | Selected Answer-drop | Latency | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Frozen Top-5 | Original 3,000 | reference | 0% | 0% | 140.88 ms | Exact baseline |
| CrossEncoder-Top5 | Original 3,000 | +0.0128 | 100% reranked | -- | 149.90 ms | Higher SP/Joint, lower Answer |
| Full | Original 3,000 | +0.0064 | 25.8% modified | 7.75% | 213.48 ms | Answer-oriented selective point |
| Lite | Revision 3,405 | -0.0063 vs Full | -- | -- | 143.97 ms | Cheaper; NI failed |
| RECOMP-660 | Original 3,000 | -0.0033 vs baseline | 100% compressed | -- | 169.64 ms | Budget control; p=0.4172 |

Full runs its generator and selector for every query even though it modifies only approximately 26% of contexts. It is selective in context modification, not in whether computation is executed. Full adds 72.60 ms/query over baseline, a 1.52x ratio, and is 63.58 ms slower than CrossEncoder-Top5. All evaluated online systems make one final answer-reader call; candidate reader outcomes are offline supervision.

Lite nearly restores baseline latency but fails the pre-frozen 0.002 Joint-F1 non-inferiority test on the revision holdout. RECOMP-660 uses the same Top-5 input, reader, support predictor, and approximately matched context budget, but its structural action space differs and its Joint change is non-significant. Pair-score pruning provides little latency reduction because semantic feature computation, rather than the number of retained pair actions, dominates generator cost: reducing k from 10 to 3 changes the component-scaled estimate only from 213.48 to 212.04 ms/query. The complete pruning sensitivity remains in the supplement and no pruned method is promoted.

Latency uses one GPU, batch size one, 50 warmup queries, and 500 measured queries with CUDA synchronization. Model loading and upstream retrieval are excluded. These measurements characterize one post-retrieval setup, not throughput, energy, mobile hardware, or a production service-level guarantee.
"""


def external_boundary() -> str:
    return """## 7. External Boundary

Frozen transfer to 1,000 2Wiki queries changes Answer/SP/Joint F1 by +0.0086/-0.0006/+0.0033; all aggregate changes are non-significant. Selected Answer-drop is 6.92%. Few-shot gate calibration reduces coverage but still reaches 5.10% Answer-drop, missing the pre-specified 4% target, so no further target-outcome tuning is performed.

Type-level analysis uses the official 2Wiki reasoning field, and no subgroup survives Benjamini-Hochberg correction. The official reasoning taxonomy therefore does not explain the aggregate transfer uncertainty after multiplicity correction. Feature diagnostics show shifts in lexical, entity, candidate, and score distributions, but these associations do not identify a root cause or a corrective mechanism.

Future analysis should test mechanism-aligned groupings, such as explicit bridge availability and answer-anchor dependence, under a separately frozen protocol. The present evidence does not establish cross-domain reliability or subgroup transfer.
"""


def limitations() -> str:
    return """## 8. Limitations and Ethical Considerations

The population effects are small, and risk control is empirical rather than certifying: 7.75%-7.83% of selected actions lower Answer F1 and 14.19%-14.86% lower Joint F1. The action-set oracle is outcome-aware and unattainable online. Its large gap is evidence of selector regret, not a prediction that the deployed policy can reach oracle performance. CrossEncoder-Top5 recovers more SP/Joint gain at lower latency than Full, which limits the incremental claim attributable to pair-complementary construction.

Both primary holdouts come from HotpotQA distractor validation. Disjointness and freezing establish same-source replication, not domain generalization. Frozen 2Wiki transfer is non-significant and calibration misses its risk target. The evaluated candidate pool contains approximately ten documents; Full scores ten pairs per query. Pair pruning does not resolve larger-pool complexity, and no claim is made about corpus-scale retrieval, changing indexes, or web search.

Full costs 213.48 ms/query on the measured A100 setup and runs generator and selector computation for every query. Lite's non-inferiority test fails, historical offline GPU-hour totals are unavailable, and no energy or alternative-hardware profile is reported. The work therefore makes no low-overhead, edge, privacy, or distributed-client claim.

The method rearranges retrieved passages and cannot recover evidence absent from the pool. Risk estimates or support predictions may behave differently across question and entity groups. Consequential use would require target-domain calibration, subgroup auditing, and explicit tolerance choices for Answer, evidence, latency, and intervention harm.
"""


def oracle_compact() -> str:
    return """# Compact Oracle Analysis

| Split | No positive action | Positive action missed | Positive action selected |
| --- | ---: | ---: | ---: |
| Original holdout (3,000) | 2,316 | 465 | 219 |
| Revision holdout (3,405) | 2,638 | 515 | 252 |

The retrospective oracle remains substantially above the frozen policy, indicating selector regret within the generated action set. It inspects target-query reader outcomes and is therefore a post-hoc diagnostic, not a deployable system or confirmatory baseline. The decomposition also shows a separate candidate-availability limitation: many queries have no positive action under the frozen training definition.

## Ratio-reference decision

Preservation-head-only, utility-head-only, and fixed-score decision variants are not added. The frozen artifacts do not store a complete pre-threshold alternative-decision protocol at the same coverage for all three variants; constructing one would require new decision rules or threshold choices after holdout access. The requested reference analysis is therefore skipped rather than tuned post hoc.
"""


def two_wiki_rewrite() -> str:
    return """# 2Wiki Boundary Rewrite

Frozen Full changes Answer/SP/Joint F1 by +0.0086/-0.0006/+0.0033 on 1,000 2Wiki queries; all aggregate effects are non-significant. No official reasoning-type subgroup survives Benjamini-Hochberg correction, and few-shot calibration reaches 5.10% Answer-drop rather than the pre-specified 4% target.

**Main-text wording:** The official reasoning taxonomy does not explain the aggregate transfer uncertainty after multiplicity correction. Transfer failure is associated with shifts in lexical, entity, candidate, and score distributions, but the analysis does not establish causality or identify a corrective mechanism.

**Future-work wording:** Future analysis should test mechanism-aligned groupings, such as explicit bridge availability and answer-anchor dependence, under a separately frozen protocol.

The paper must not claim that the method transfers to compositional or bridge-like questions, that lexical/entity shift is the root cause, or that further target calibration would establish generalization.
"""


def rebuttal() -> str:
    return """# Final SIGIR-AP Rebuttal Templates

## Why use Full if CrossEncoder has higher Joint F1?

We agree that CrossEncoder-Top5 obtains higher SP and Joint F1 under the matched protocol. It also lowers Answer F1 below the frozen baseline by 0.0105/0.0066, whereas Full improves Answer and Joint on both holdouts. Full costs more. We therefore do not claim universal superiority: the systems occupy different evaluated answer-evidence-latency operating points, and neither dominates all reported objectives.

## Is Answer F1 more important than Joint F1?

No universal metric hierarchy is assumed. We report Answer, SP, and Joint together because they expose different behavior. Full's training objective is oriented toward preserving answer quality while improving joint utility, but an application may prefer the stronger evidence retrieval of CrossEncoder. The paper characterizes this preference rather than resolving it.

## Does the oracle prove that the selector is poor?

Large selector regret is an explicit finding, but the oracle reads target-query reader outcomes and is unattainable online. It is restricted to the frozen action set and separates two limitations: queries without a training-positive action and queries where such an action exists but the policy misses it. We do not claim selector optimality or deployable oracle performance.

## Is pair complementarity necessary?

Removing pair complementarity or bounded chains reduces development opportunity metrics, supporting their role in the frozen action generator. However, strong independent relevance ranking recovers or exceeds Full's SP/Joint result. Our claim is therefore bounded to expanded action opportunity and an Answer-oriented selective operating point, not unique superiority of pair features.

## Is Full too expensive?

Full costs 213.48 ms/query, 1.52x the 140.88-ms baseline and more than the 149.90-ms CrossEncoder. The generator and selector run for every query even though only about 26% of contexts are modified. Lite nearly restores baseline latency but fails the pre-frozen non-inferiority test. We report this cost boundary rather than describing Full as efficient.

## Does 2Wiki demonstrate no generalization?

The aggregate transfer result is non-significant, no official reasoning-type subgroup survives FDR correction, and few-shot calibration misses the 4% Answer-drop target. We therefore make no cross-domain reliability claim. The available taxonomy also does not explain the uncertainty, so further mechanism-aligned analysis requires a separately frozen protocol.
"""


def build() -> dict[str, int]:
    source = SOURCE_PAPER.read_text(encoding="utf-8")
    method_protocol = section(source, "## 2. Related Work", "## 5. Main Results")
    disagreement = disagreement_values()
    paper = "\n\n".join([
        f"# {TITLE}",
        ABSTRACT,
        INTRODUCTION.rstrip(),
        method_protocol,
        main_results(disagreement).rstrip(),
        mechanism_and_cost().rstrip(),
        external_boundary().rstrip(),
        limitations().rstrip(),
        CONCLUSION.rstrip(),
    ]) + "\n"

    supplement = SOURCE_SUPPLEMENT.read_text(encoding="utf-8").rstrip()
    supplement = supplement.replace("# Appendix", "# Final Supplement")
    supplement = supplement.replace("Answer-safe rate", "Training-label preservation rate")
    supplement = supplement.replace("answer-safe title-utility", "training-positive title-utility")
    supplement = supplement.replace("original training-positive title-utility training definition", "original training-positive title-utility criterion")
    supplement = supplement.replace("Best few-shot gate", "Lowest-Answer-drop evaluated few-shot gate")
    disagreement_text = (HERE / "crossencoder_full_disagreement_analysis.md").read_text(encoding="utf-8")
    disagreement_text = disagreement_text.split("\n", 1)[1].strip()
    supplement += "\n\n## V. CrossEncoder-Full Disagreement Details\n\n" + disagreement_text
    supplement += "\n\n## W. Answer-Joint-Latency Figure Data\n\n"
    supplement += "The figure uses only frozen points available for each split: RECOMP-660 is shown only for the original holdout and Lite only for the revision holdout. Non-dominance is assessed only among the evaluated systems, metrics, and measured post-retrieval latency. Source values are in `outputs/answer_joint_latency_points.csv`.\n"

    artifacts = {
        "title_sigirap_final.md": f"# {TITLE}\n",
        "abstract_sigirap_final.md": "# Abstract\n\n" + ABSTRACT + "\n",
        "introduction_sigirap_final.md": INTRODUCTION + "\n",
        "conclusion_sigirap_final.md": CONCLUSION + "\n",
        "paper_sigirap_final_9page.md": paper,
        "paper_sigirap_final_supplement.md": supplement + "\n",
        "oracle_compact_analysis.md": oracle_compact(),
        "two_wiki_boundary_rewrite.md": two_wiki_rewrite(),
        "rebuttal_sigirap_final.md": rebuttal(),
    }
    for name, text in artifacts.items():
        (HERE / name).write_text(text, encoding="utf-8")
    return {name: len(re.findall(r"\b\w+\b", text)) for name, text in artifacts.items()}


def main() -> None:
    counts = build()
    print(json.dumps({"status": "complete", "word_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
