#!/usr/bin/env python3
"""Build the review-polished V5 paper without changing frozen evidence."""

from pathlib import Path
import shutil
from typing import Optional


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "v5_final_review_optimized_submission"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"Missing section boundary: {start!r} -> {end!r}")
    prefix, rest = text.split(start, 1)
    _, suffix = rest.split(end, 1)
    return prefix + replacement.rstrip() + "\n\n" + end + suffix


def extract_between(text: str, start: str, end: Optional[str]) -> str:
    if start not in text:
        raise ValueError(f"Missing extraction start: {start!r}")
    rest = text.split(start, 1)[1]
    if end is not None:
        rest = rest.split(end, 1)[0]
    return start + rest.rstrip() + "\n"


abstract = """## Abstract

Multi-hop question answering depends not only on retrieving relevant documents but also on constructing an ordered context that exposes complementary evidence without displacing answer-bearing passages. We study the **candidate-opportunity gap**: a selector cannot repair a context when its candidate actions omit the needed evidence combination. Our method scores pair complementarity, constructs bounded two-document chains, preserves baseline anchors, and uses fully nested selective intervention with exact fallback. **Reader-safe** denotes an answer-preservation-oriented, risk-controlled selection objective; it does not provide a per-query harm guarantee. On frozen 3,000- and 3,405-query HotpotQA holdouts, the method improves Answer, supporting-fact (SP), and Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080, respectively. The policy intervenes on 25.8% and 25.9% of queries. On the first holdout, selected-query mean deltas are +0.0340/+0.0219/+0.0250, but most interventions tie the baseline and 7.75% reduce Answer F1; these are conditional descriptive results, not causal effects. Measured post-retrieval latency is 213.5 ms/query versus 140.9 ms/query for frozen Top-5, with one final reader call in both. Frozen 2Wiki transfer is non-significant, and few-shot calibration misses its pre-specified answer-risk target. The results support a bounded quality-risk-cost trade-off, not broad efficiency, safety, or cross-dataset claims.
"""


introduction = """## 1. Introduction

Multi-hop question answering (QA) is often described as finding relevant documents, yet a reader consumes an ordered and budget-limited context. A usable context must expose complementary hops, retain the passage that gives the answer its lexical form, and place evidence where the reader can use it. Adding a relevant document can therefore help support recovery while simultaneously displacing an answer anchor.

This creates a structural limit for post-retrieval selection. A selector chooses only among contexts proposed by its action generator. If no proposal contains a reader-compatible repair, a better selector cannot help. We call this mismatch the **candidate-opportunity gap**. An initial fixed-action study and a later heuristic expansion showed that increasing the number of isolated insertions and replacements did not reliably increase the density of actions that improve downstream reader outcomes without reducing answer quality.

We address this gap with a Full Pair-Complementary Action Generator. It models whether two documents supply different parts of a multi-hop chain, builds bounded two-document actions, and protects high-value passages from the original Top-5 context. The implementation combines lexical, MPNet, and cross-encoder signals with missing-hop, document-opportunity, and pair-complementarity models. These components define the frozen Full recipe; our claim concerns their joint system, not a monotonic benefit from every feature.

A separate two-head selector estimates answer preservation and positive reader utility. Frozen thresholds and a coverage budget permit an intervention only when both conditions pass; otherwise the system returns the original Top-5 context exactly. The reader runs once on the final context. This is an answer-preservation-oriented, risk-controlled objective, not a certification that every selected action is harmless.

We train and evaluate the system with a fully nested five-fold protocol. Generator and selector models fit outer-training queries, thresholds are derived from inner out-of-fold predictions, and outer-test outcomes are never used for architecture or threshold selection. The frozen system is then evaluated on two disjoint same-source HotpotQA holdouts containing 3,000 and 3,405 queries. Both show modest positive population changes in Answer, SP, and Joint F1. Larger means on the roughly one quarter of queries selected for intervention are reported separately with their wins, losses, ties, and drop rates.

The evidence also establishes practical boundaries. Full costs 1.52 times the measured post-retrieval latency of frozen Top-5. A review-driven Lite simplification fails an independently frozen non-inferiority criterion. A budget-controlled RECOMP comparison does not establish a general method ranking. Frozen transfer to 2Wiki is non-significant, and target-domain calibration does not meet its answer-risk target. The evaluated candidate pool is the bounded HotpotQA distractor set, not a corpus-scale retrieval system.

Our contributions are:

1. We formulate the candidate-opportunity gap for reader-aware multi-hop context construction.
2. We introduce pair-complementary, anchor-preserving action generation with bounded two-document chains.
3. We combine the generator with fully nested, risk-controlled selective intervention and exact fallback.
4. We separate population effects from policy-conditional effects and report quality, risk, latency, comparison, reader, and transfer boundaries under frozen protocols.
"""


selected_results = """### 5.2 Descriptive Effects on Policy-Selected Interventions

Population and conditional views answer different questions. The population rows describe the effect of running the frozen policy on every query. The selected rows describe only the contexts that the policy actually changed. They must therefore be interpreted together.

| Holdout | Metric | Population delta | Coverage | Selected mean | Wins/Losses/Ties | Drop rate | Median [Q25, Q75] |
|---|---|---:|---:|---:|---:|---:|---:|
| Original 3,000 | Answer F1 | +0.0088 | 774/3000 (25.8%) | +0.0340 | 89/60/625 | 7.75% | 0 [0, 0] |
| Original 3,000 | SP F1 | +0.0056 | 774/3000 (25.8%) | +0.0219 | 123/100/551 | 12.92% | 0 [0, 0] |
| Original 3,000 | Joint F1 | +0.0064 | 774/3000 (25.8%) | +0.0250 | 141/115/518 | 14.86% | 0 [0, 0] |
| Revision 3,405 | Answer F1 | +0.0116 | 881/3405 (25.9%) | +0.0447 | 107/69/705 | 7.83% | 0 [0, 0] |
| Revision 3,405 | SP F1 | +0.0061 | 881/3405 (25.9%) | +0.0237 | 127/94/660 | 10.67% | 0 [0, 0] |
| Revision 3,405 | Joint F1 | +0.0080 | 881/3405 (25.9%) | +0.0309 | 169/125/587 | 14.19% | 0 [0, 0] |

The zero medians and interquartile ranges show that most selected contexts tie the baseline. Answer F1 decreases on 60 of 774 original-holdout interventions and 69 of 881 revision-holdout interventions; Joint F1 decreases more often. In both holdouts, every fallback context and metric is exactly identical to the baseline. Although the selected subset has larger mean deltas, most selected contexts tie the baseline and some are harmful; the conditional result characterizes the policy's chosen subset rather than an oracle-improvable population.
"""


candidate_scope = """### 7.1 Candidate-Pool Scope and Pair Complexity

The method begins after retrieval from the official HotpotQA distractor pool, which contains approximately ten documents per query. In the 3,000-query holdout, 2,973 queries have at least ten available documents, only one has at least twenty, and none has fifty or one hundred. There is therefore no common fixed subset on which to claim large-pool scaling.

Pair construction over a retained set of size $L$ is quadratic before pruning, with at most $L(L-1)/2$ pairs. The frozen protocol sets $L=10$, which gives 45 possible pairs before pruning; the measured deployment scores ten pairs per query. These constants bound the reported latency but do not demonstrate corpus-scale behavior. Future tests would require a separately frozen protocol for subquadratic candidate pairing, approximate nearest-neighbor pair retrieval, adaptive Top-$L$, and calibration under continuously changing indexes.
"""


analysis = """## 9. Analysis

**Opportunity before selection.** The action generator determines whether repair is possible at all. Pair complementarity raises the chance that a proposal contains both hops, while bounded construction prevents opportunity from becoming an uncontrolled permutation search. Selection then trades coverage for answer risk. This explains why conditional means can exceed population means without implying a broad treatment effect.

**What the Lite failure means.** Pair complementarity, chains, anchors, and selective risk control are the most interpretable mechanisms. Yet the untouched holdout shows that lexical pair features alone do not preserve Full quality within the chosen margin. Missing-hop, MPNet, cross-encoder, and document-opportunity components therefore remain in the stronger implementation. Their mixed individual ablations support neither a claim that each is necessary nor a claim that each always helps.

**Compression versus structured action.** Equalizing token budget removes the most obvious information-volume confound, but it does not equalize objectives. Sentence packing chooses text spans; Full selects a small structural intervention while retaining five-document coverage. The comparison constrains interpretation rather than identifying one universally better constructor.

**Directional answer-reader check.** We replay the same frozen baseline and selected contexts with FLAN-T5-Large and UnifiedQA-T5-Large. FLAN Answer F1 changes from 0.6183 to 0.6271 (+0.0088), while UnifiedQA changes from 0.5662 to 0.5772 (+0.0110). Their Joint F1 point estimates change from 0.3292 to 0.3356 (+0.0064) and from 0.3045 to 0.3130 (+0.0085). Both rows reuse the same support predictor, whose SP F1 changes from 0.4930 to 0.4987 (+0.0056). The second answer reader therefore supplies directional evidence for answer behavior only. It is not an independent SP replication, and the Joint direction is not independent of the shared support component.

**Transfer as a gate boundary.** 2Wiki retains positive Answer and Joint point estimates, but all three tests are non-significant and the Hotpot risk scores are misaligned with target-domain harm. Target-train calibration lowers selected answer-drop only partially. The evidence therefore separates reusable action construction from unresolved risk calibration under shift.
"""


limitations = """## 10. Limitations and Ethical Considerations

1. **Small population effects and added latency.** Full changes Answer/SP/Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 on the two holdouts, while measured post-retrieval latency rises from 140.88 to 213.48 ms/query (1.52x). The evidence supports a quality-risk-cost trade-off, not a broad efficiency claim.

2. **Risk control is not a per-query guarantee.** Reader-safe is an objective label for answer-preservation-oriented selection, not a per-query guarantee. Among selected interventions, 7.75% and 7.83% reduce Answer F1, and 14.86% and 14.19% reduce Joint F1. The selector reduces average risk but cannot guarantee that an individual action will help or tie.

3. **Both confirmatory sets are same-source.** The 3,000- and 3,405-query holdouts are disjoint from development and from each other, but both come from HotpotQA distractor validation. They establish frozen same-source replication, not domain generalization.

4. **External transfer fails its planned criterion.** On 2Wiki, the frozen deltas are non-significant and few-shot calibration reaches a 5.10% selected answer-drop rate rather than the pre-specified 4% target. Cross-dataset risk calibration remains unresolved and requires labeled target-domain reader outcomes.

5. **The candidate pool is bounded.** The study starts from roughly ten Hotpot distractor documents. Pair construction is quadratic in retained set size before pruning, although the frozen system scores ten pairs per query. Corpus-scale retrieval and changing-index behavior are not evaluated.

6. **Support replication is shared.** UnifiedQA changes the answer reader while reusing the same selected contexts and support predictor. It provides directional answer-reader evidence, not independent SP replication; its Joint result also contains the shared support component.

7. **The Lite simplification fails non-inferiority.** Lite reduces measured latency to 143.97 ms/query but is 0.0063 Joint F1 below Full on the independent holdout, with a 95% interval entirely beyond the 0.002 non-inferiority margin. The semantic Full recipe therefore remains necessary for the reported result.

8. **Historical offline cost is incomplete.** The online benchmark is reproducible on one A100 with batch size one, but historical GPU-hour totals for offline outcome labeling and fold-specific training were not recorded. We do not reconstruct them retrospectively.

The method rearranges supplied passages rather than generating evidence. This improves traceability but cannot recover facts absent from the pool. Errors from the fixed answer readers or support predictor may also vary by entity, language, or question type, so deployment in consequential settings requires direct auditing beyond aggregate benchmarks.
"""


conclusion = """## 11. Conclusion

Multi-hop context selection is limited by the actions its generator exposes. Pair-complementary construction creates bounded two-document alternatives, preserves baseline answer anchors, and submits an action to a fully nested, risk-controlled selector with exact fallback. Two frozen same-source HotpotQA holdouts show modest positive population changes, while policy-selected queries have larger mean changes but mostly tie the baseline and include measurable harm. Full also costs 1.52 times the measured post-retrieval latency. The failed Lite non-inferiority test, non-significant 2Wiki transfer, bounded candidate pool, and shared support predictor define the present scope. The contribution is therefore a controlled method and evaluation for improving context opportunity under a fixed reader, together with an explicit account of when its quality gains do and do not justify intervention.
"""


appendix_extra = """
## H. Multi-Reader Supporting Analysis

| Answer reader | Baseline Answer F1 | Selected Answer F1 | Answer delta | Baseline SP F1 | Selected SP F1 | SP delta | Baseline Joint F1 | Selected Joint F1 | Joint delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLAN-T5-Large | 0.6183 | 0.6271 | +0.0088 | 0.4930* | 0.4987* | +0.0056* | 0.3292 | 0.3356 | +0.0064 |
| UnifiedQA-T5-Large | 0.5662 | 0.5772 | +0.0110 | 0.4930* | 0.4987* | +0.0056* | 0.3045 | 0.3130 | +0.0085 |

The same frozen contexts are replayed for both answer readers. Asterisks mark values from one shared support predictor. The positive Answer F1 direction is therefore the only independently reader-varying observation. The SP values are repeated rather than replicated, and Joint F1 combines each answer reader with that shared support component. We consequently describe this as a directional answer-reader check, not two independent end-to-end reader pipelines.

## I. Candidate-Pool Scope

| Available documents in the 3,000-query holdout | Queries meeting threshold |
|---:|---:|
| At least 10 | 2,973 |
| At least 20 | 1 |
| At least 50 | 0 |
| At least 100 | 0 |

The official distractor pool is approximately ten documents per query. With retained size $L=10$, exhaustive pair formation would create 45 pairs before pruning; the frozen implementation scores ten pairs per query. The reported benchmark is therefore a bounded post-retrieval context-construction test. It does not measure corpus-scale candidate generation, adaptive large-$L$ behavior, or continuously updated indexes.

Potential extensions include subquadratic pair proposals, approximate nearest-neighbor retrieval over pair representations, adaptive Top-$L$ allocation, and risk calibration under changing candidate distributions. Each would require a new frozen protocol rather than extrapolation from the current timing result.
"""


def main() -> None:
    main_text = (SOURCE / "paper_anonymous_v5_final.md").read_text()
    main_text = main_text.replace(
        "# Pair-Complementary Context Construction with Reader-Safe Selection for Multi-Hop QA",
        "# Pair-Complementary Context Construction with Risk-Controlled Selection for Multi-Hop QA",
        1,
    )
    main_text = replace_between(main_text, "## Abstract", "## 1. Introduction", abstract)
    main_text = replace_between(main_text, "## 1. Introduction", "## 2. Related Work", introduction)
    main_text = main_text.replace(
        "During training only, an action is answer-safe when the frozen reader's Answer F1 is no lower than on $C_0$ and positive when it improves the reader-side joint answer-evidence utility. Query opportunity is the existence of at least one safe positive action in $A(q)$. It is an empirical upper bound for any selector restricted to that set, not an inference-time label.",
        "During training only, an action receives an answer-preservation label when the frozen reader's Answer F1 is no lower than on $C_0$, and a positive-utility label when it improves the reader-side joint answer-evidence utility. Query opportunity is the existence of at least one action satisfying both labels in $A(q)$. It is an empirical upper bound for any selector restricted to that set, not an inference-time label or guarantee.",
    )
    main_text = main_text.replace("### 3.3 Reader-Safe Selector", "### 3.3 Risk-Controlled Selector")
    main_text = main_text.replace(
        "We distinguish action opportunity from action selection: a reader-safe selector remains limited by the combinations exposed by its generator.",
        "We distinguish action opportunity from action selection: a risk-controlled selector remains limited by the combinations exposed by its generator.",
    )
    main_text = main_text.replace(
        "The selector has two balanced logistic heads. The safety head estimates whether an action preserves baseline answer quality; the utility head estimates whether it yields a positive reader-side change. Inner out-of-fold predictions determine both thresholds and a 10-30% coverage budget. On an outer-test query, an action is eligible only if it passes both heads. The highest-ranked eligible action is applied while the fold-level budget remains; all other queries receive the exact baseline. The reader then evaluates one final context.",
        "The selector has two balanced logistic heads. The preservation head estimates whether an action retains baseline answer quality; the utility head estimates whether it yields a positive reader-side change. Inner out-of-fold predictions determine both thresholds and a 10-30% coverage budget. On an outer-test query, an action is eligible only if it passes both heads. The highest-ranked eligible action is applied while the fold-level budget remains; all other queries receive the exact baseline. The reader then evaluates one final context. These estimates control average intervention risk but do not certify individual actions.",
    )
    main_text = replace_between(
        main_text,
        "### 5.2 Descriptive Effects on Policy-Selected Interventions",
        "### 5.3 Full-to-Lite Non-Inferiority",
        selected_results,
    )
    main_text = main_text.replace(
        "The intended deployment is auditable context organization over a bounded post-retrieval pool. Corpus-scale indexing, dynamic web search, and production streaming are outside the evaluation.",
        "The intended deployment is auditable context organization over a bounded post-retrieval pool. Corpus-scale indexing, dynamic web search, and continuously updated production indexes are outside the evaluation.\n\n" + candidate_scope,
    )
    main_text = replace_between(main_text, "## 9. Analysis", "## 10. Limitations and Ethical Considerations", analysis)
    main_text = replace_between(main_text, "## 10. Limitations and Ethical Considerations", "## 11. Conclusion", limitations)
    main_text = main_text.split("## 11. Conclusion", 1)[0] + conclusion
    main_text = main_text.rstrip() + "\n"

    appendix = (SOURCE / "paper_appendix_v5_final.md").read_text().replace(
        "anchor preservation, and selective safety",
        "anchor preservation, and selective risk control",
    ).rstrip() + "\n" + appendix_extra
    full = main_text.rstrip() + "\n\n" + appendix.rstrip() + "\n"

    (ROOT / "paper_anonymous_review_polished.md").write_text(main_text)
    (ROOT / "paper_full_review_polished.md").write_text(full)
    (ROOT / "paper_appendix_review_polished.md").write_text(appendix)

    sections = {
        "abstract_review_polished.md": extract_between(main_text, "## Abstract", "## 1. Introduction"),
        "introduction_review_polished.md": extract_between(main_text, "## 1. Introduction", "## 2. Related Work"),
        "results_review_polished.md": extract_between(main_text, "## 5. Main Results", "## 6. Budget-Matched Compression"),
        "limitations_review_polished.md": extract_between(main_text, "## 10. Limitations and Ethical Considerations", "## 11. Conclusion"),
    }
    for name, content in sections.items():
        (ROOT / name).write_text(content)

    shutil.copy2(SOURCE / "references.bib", ROOT / "references.bib")


if __name__ == "__main__":
    main()
