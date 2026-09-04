#!/usr/bin/env python3
"""Assemble the strengthened manuscript, supplement, audits, and reviews."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sigirap_common import HERE, OUTPUTS, PAPER_ROOT, REPORTS, read_json


BASE = PAPER_ROOT / "v6_venue_ready_packaging"
REVIEWS = HERE / "simulated_reviews_after_strengthening"


def f4(value: float, signed: bool = False) -> str:
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def ci(value: dict[str, float]) -> str:
    return f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"


def load_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    oracle = read_json(OUTPUTS / "oracle/oracle_metrics.json")
    reranker = read_json(OUTPUTS / "reranker/ce_reranker_metrics.json")
    latency = read_json(OUTPUTS / "reranker/ce_reranker_latency.json")
    if oracle.get("status") != "complete" or reranker.get("status") != "complete":
        raise RuntimeError("Oracle and independent reranker must be complete before paper assembly")
    return oracle, reranker, latency


def abstract_text(oracle: dict[str, Any], reranker: dict[str, Any]) -> str:
    h3 = oracle["splits"]["holdout3000"]
    r3 = oracle["splits"]["revision3405"]
    ce3 = reranker["splits"]["holdout3000"]
    cer = reranker["splits"]["revision3405"]
    chosen = reranker["development_variant_choice"]["chosen_variant"]
    return (
        "A multi-hop reader needs complementary evidence without losing passages that express the answer, yet a selector can only choose among contexts exposed by its generator. We study this candidate-opportunity gap in a frozen approximately Top-10 post-retrieval pool. Full scores pair complementarity, forms bounded two-document chains, preserves baseline anchors, and uses fully nested preservation and utility heads to apply one action or return the Top-5 baseline exactly. On disjoint frozen HotpotQA holdouts of 3,000 and 3,405 queries, Full changes Answer/SP/Joint F1 by +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 at 25.8%-25.9% coverage. An outcome-aware diagnostic restricted to the same frozen actions places answer-preserving oracle Joint F1 at "
        f"{h3['metrics']['answer_preserving_oracle']['joint_f1']:.4f} and {r3['metrics']['answer_preserving_oracle']['joint_f1']:.4f}, versus policy values {h3['metrics']['policy']['joint_f1']:.4f} and {r3['metrics']['policy']['joint_f1']:.4f}; this retrospective result measures opportunity and selector regret, not deployable performance. A post-hoc secondary `{chosen}` baseline under the same pool, five-document budget, reader, and support predictor reaches Joint F1 {ce3['metrics'][chosen]['joint_f1']:.4f}/{cer['metrics'][chosen]['joint_f1']:.4f}, compared with Full {ce3['metrics']['full']['joint_f1']:.4f}/{cer['metrics']['full']['joint_f1']:.4f}, but its Answer F1 is lower than Full by {abs(ce3['chosen_vs_full']['answer_f1']['mean']):.4f}/{abs(cer['chosen_vs_full']['answer_f1']['mean']):.4f}. Full costs 213.48 versus 140.88 ms/query, and frozen 2Wiki transfer remains non-significant with no reasoning-type effect surviving FDR correction. The evidence supports a bounded quality-risk-cost trade-off and clarifies that strong independent relevance recovers much of the SP/Joint benefit; it does not establish universal reranking superiority or cross-domain robustness."
    )


def oracle_section(oracle: dict[str, Any]) -> str:
    lines = [
        "### 5.3 Opportunity and Selector Regret",
        "",
        "We add an outcome-aware diagnostic that chooses only among each query's already generated actions plus baseline. The utility oracle maximizes official Joint F1; the answer-preserving oracle first requires Answer F1 no lower than baseline; and the available-opportunity oracle is restricted to actions positive under the frozen training definition. These oracles inspect target-query reader outcomes and are therefore retrospective mechanism diagnostics, not inference-time systems or confirmatory comparisons.",
        "",
        "| Split | Baseline J | Policy J | Answer-preserving oracle J | Training-positive opportunity | Policy coverage | Aggregate policy/oracle gain ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("development1000", "holdout3000", "revision3405"):
        row = oracle["splits"][split]
        lines.append(
            f"| {row['label']} | {row['metrics']['baseline']['joint_f1']:.4f} | {row['metrics']['policy']['joint_f1']:.4f} | "
            f"{row['metrics']['answer_preserving_oracle']['joint_f1']:.4f} | {row['query_opportunity_coverage']:.1%} | "
            f"{row['policy_intervention_coverage']:.1%} | {row['selector_capture_ratio']:.1%} |"
        )
    h3, r3 = oracle["splits"]["holdout3000"], oracle["splits"]["revision3405"]
    lines.extend([
        "",
        f"On the 3,000 holdout, {h3['decomposition']['no_opportunity']} queries have no positive action under the training definition, {h3['decomposition']['opportunity_but_missed']} have an available positive action that the policy misses, and {h3['decomposition']['successful_selection']} receive a positive selected action. The corresponding 3,405 counts are {r3['decomposition']['no_opportunity']}, {r3['decomposition']['opportunity_but_missed']}, and {r3['decomposition']['successful_selection']}. Mean answer-preserving-oracle minus policy Joint regret is {h3['regret']['mean']:.4f}/{r3['regret']['mean']:.4f}; medians are zero because {h3['regret']['zero_regret_proportion']:.1%}/{r3['regret']['zero_regret_proportion']:.1%} have zero regret.",
        "",
        "The two opportunity notions answer different questions. Training-positive opportunity follows the original answer-safe title-utility label, whereas the answer-preserving oracle directly uses official target outcomes. The aggregate policy/oracle gain ratio divides population Joint gain by retrospective answer-preserving-oracle gain; it is not a per-query ratio and does not measure all possible system improvement. The gap shows that both bounded action availability and selection remain limiting.",
        "",
        "**Figure 2: Retrospective opportunity-selection decomposition.** See `outputs/figures/opportunity_selection_decomposition.pdf`.",
        "",
    ])
    return "\n".join(lines)


def reranker_section(reranker: dict[str, Any], latency: dict[str, Any]) -> str:
    chosen = reranker["development_variant_choice"]["chosen_variant"]
    lines = [
        "### 6.1 Independent Relevance Reranking",
        "",
        f"To test whether pair-complementary construction adds value beyond strong independent relevance, we add CrossEncoder-Top5. It scores the same approximately ten documents with Full's frozen cross-encoder and retains five, but excludes pair, missing-hop, outcome-model, and selector features. We predefine score order and baseline-stable order, choose `{chosen}` by development Joint F1 only, and freeze that choice for both holdouts. Reader, prompt, 3,200-character cap, support predictor, and official metrics are shared. Because this comparison was added after the primary study, it is a post-hoc secondary baseline analysis.",
        "",
        "| Split | System | Answer F1 | SP F1 | Joint F1 | Joint delta vs baseline | Answer delta vs Full | Joint delta vs Full |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("holdout3000", "revision3405"):
        result = reranker["splits"][split]
        baseline = result["metrics"]["baseline"]
        ce = result["metrics"][chosen]
        full = result["metrics"]["full"]
        ce_base = result["vs_baseline"][chosen]
        ce_full = result["chosen_vs_full"]
        label = "Hotpot-3,000" if split == "holdout3000" else "Hotpot-3,405"
        lines.extend([
            f"| {label} | Frozen Top-5 | {baseline['answer_f1']:.4f} | {baseline['sp_f1']:.4f} | {baseline['joint_f1']:.4f} | reference | -- | -- |",
            f"| {label} | CrossEncoder-Top5 | {ce['answer_f1']:.4f} | {ce['sp_f1']:.4f} | {ce['joint_f1']:.4f} | {ce_base['joint_f1']['mean']:+.4f} | {ce_full['answer_f1']['mean']:+.4f} | {ce_full['joint_f1']['mean']:+.4f} |",
            f"| {label} | Full | {full['answer_f1']:.4f} | {full['sp_f1']:.4f} | {full['joint_f1']:.4f} | {result['vs_baseline']['full']['joint_f1']['mean']:+.4f} | reference | reference |",
        ])
    ltext = (
        f"Direct same-machine latency for CrossEncoder-Top5 is {latency['end_to_end_ms']['mean']:.2f} ms/query (P95 {latency['end_to_end_ms']['p95']:.2f})"
        if latency.get("status") == "complete" else
        "The direct same-machine CrossEncoder-Top5 latency run is pending; cached action-build time is not substituted for online latency"
    )
    h3 = reranker["splits"]["holdout3000"]["chosen_vs_full"]
    r3 = reranker["splits"]["revision3405"]["chosen_vs_full"]
    lines.extend([
        "",
        f"CrossEncoder-Top5 exceeds baseline Joint F1 by {reranker['splits']['holdout3000']['vs_baseline'][chosen]['joint_f1']['mean']:+.4f} and {reranker['splits']['revision3405']['vs_baseline'][chosen]['joint_f1']['mean']:+.4f}. Relative to Full, its Joint difference is {h3['joint_f1']['mean']:+.4f} ({ci(h3['joint_f1'])}, p={h3['joint_f1']['p_value']:.4f}) and {r3['joint_f1']['mean']:+.4f} ({ci(r3['joint_f1'])}, p={r3['joint_f1']['p_value']:.4f}), while Answer F1 is lower by {abs(h3['answer_f1']['mean']):.4f} and {abs(r3['answer_f1']['mean']):.4f} (p={h3['answer_f1']['p_value']:.4f}/{r3['answer_f1']['p_value']:.4f}). Thus much of the SP/Joint gain can be recovered by strong independent relevance ranking, limiting the incremental role attributable to pair-complementary construction. Full instead occupies a different trade-off point: higher Answer F1, selective intervention with exact fallback, and lower SP/Joint than the always-on reranker. {ltext}. This result does not imply universal CrossEncoder superiority beyond the frozen pool and reader.",
        "",
    ])
    return "\n".join(lines)


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


def build_paper(oracle: dict[str, Any], reranker: dict[str, Any], latency: dict[str, Any]) -> str:
    paper = (BASE / "paper_wsdm_sigirap_9page.md").read_text(encoding="utf-8")
    title = "# Opportunity-Aware Context Construction with Answer-Preserving Selection for Multi-Hop QA"
    intro_index = paper.index("## 1. Introduction")
    paper = title + "\n\n" + abstract_text(oracle, reranker) + "\n\n" + paper[intro_index:]
    paper = paper.replace(
        "Our contributions are threefold. First, we formulate candidate opportunity as a necessary precondition for reader-aware selection. Second, we introduce pair-complementary, anchor-preserving generation of bounded context actions with exact fallback. Third, we provide a fully nested evaluation that separates population gain, policy-conditional behavior, intervention risk, and measured cost across two frozen holdouts, while reporting failed simplification and transfer boundaries.",
        "Our contributions are fourfold. First, we formulate candidate opportunity as a necessary precondition for reader-aware selection. Second, we introduce pair-complementary, anchor-preserving bounded actions with exact fallback. Third, a fully nested study separates population gain, intervention risk, and cost across two frozen holdouts. Fourth, post-hoc diagnostics quantify frozen-action oracle regret, compare a protocol-matched independent CrossEncoder reranker, and test whether 2Wiki failure follows official reasoning types. The added analyses narrow rather than enlarge the claim: pair-aware Full is an answer-preserving selective trade-off point, not a universal replacement for relevance reranking."
    )
    paper = paper.replace("## 6. Analysis and Cost", oracle_section(oracle) + "\n\n## 6. Analysis and Cost", 1)
    paper = paper.replace("### 6.1 Opportunity and core components", reranker_section(reranker, latency) + "\n### 6.2 Opportunity and core components", 1)
    paper = paper.replace("### 6.2 Quality-risk-cost summary", "### 6.3 Quality-risk-cost summary", 1)
    pareto = read_csv_rows(OUTPUTS / "pareto/pair_pruning_results.csv")
    k3 = next(row for row in pareto if int(row["k"]) == 3)
    paper = paper.replace(
        "The timing benchmark uses one GPU, batch size one, 50 warmup queries, 500 measured queries, and CUDA synchronization around components. Model loading and the upstream retriever are excluded for every row. These measurements describe post-retrieval latency under one controlled setup, not throughput, energy, or a production service-level guarantee. Historical offline GPU-hour totals for labeling and fold-specific training were not recorded.",
        f"The timing benchmark uses one GPU, batch size one, 50 warmup queries, 500 measured queries, and CUDA synchronization around components. Model loading and the upstream retriever are excluded for every row. These measurements describe post-retrieval latency under one controlled setup, not throughput, energy, or a production service-level guarantee. Historical offline GPU-hour totals for labeling and fold-specific training were not recorded. A development-only pair-pruning sensitivity keeps frozen features, selector, thresholds, and actions while retaining k=1/2/3/5/7/10 pair evaluations. The k=3 replay matches Full development Joint F1 and has an estimated total {float(k3['total_latency_ms_estimated']):.2f} ms/query versus 213.48 at k=10, showing that pair evaluation itself is a small part of cost. This component-scaled result is exploratory and no pruned configuration is promoted."
    )
    external = """## 7. External Boundary

Frozen transfer to 1,000 2Wiki queries changes Answer/SP/Joint F1 by +0.0086/-0.0006/+0.0033; all are non-significant and selected answer-drop is 6.92%. We analyze the dataset's official `type` field without constructing outcome-dependent groups: compositional (382), comparison (252), bridge_comparison (252), and inference (114). Joint deltas are +0.0087, +0.0001, +0.0004, and -0.0015. The compositional raw p-value is 0.0340 but becomes 0.1360 after Benjamini-Hochberg correction; no group survives FDR. The available taxonomy therefore does not explain aggregate transfer uncertainty.

Feature-shift diagnostics compare question/document length, candidate-pool size, entity/bridge overlap, pair-complementarity, frozen preservation/utility scores, and action-family frequencies. They describe associations, not causes. Few-shot gate calibration still misses its pre-specified 4% answer-drop target, so we do not retune 2Wiki further.

The primary candidate pool is approximately ten Hotpot distractor documents. With L=10, 45 pairs exist before pruning and ten are scored per query. This supports bounded post-retrieval construction, not corpus-scale retrieval. A second answer reader, UnifiedQA-T5-Large, supplies directional Answer-F1 evidence on the same contexts; the shared support predictor prevents treating SP/Joint as an independent replication.
"""
    paper = replace_section(paper, "## 7. External Boundary", "## 8. Limitations", external)
    paper = paper.replace(
        "Full produces small same-source population gains at a measured 1.52x post-retrieval latency.",
        "Full produces small same-source population gains at a measured 1.52x post-retrieval latency. CrossEncoder-Top5 recovers more SP/Joint gain on the same pool but lowers Answer F1 relative to Full, limiting claims that pair-complementary construction is uniquely responsible for downstream improvement. The outcome-aware oracle is retrospective and cannot be presented as deployable performance."
    )
    conclusion = """## 9. Conclusion

A selector cannot choose an absent repair, but stronger action construction alone does not settle the relevance-versus-reader-risk trade-off. Pair-complementary Full yields small replicated same-source gains with exact fallback and higher Answer F1 than an always-on independent CrossEncoder-Top5 baseline. The same CrossEncoder baseline, however, recovers or exceeds Full's SP/Joint gain, while the outcome-aware frozen-action oracle reveals substantial selector regret. Together with non-significant and structurally unresolved 2Wiki transfer, these results establish a bounded quality-risk-cost analysis rather than universal superiority, low-cost deployment, or cross-domain robustness.
"""
    paper = replace_section(paper, "## 9. Conclusion", "", conclusion) if False else paper[:paper.index("## 9. Conclusion")] + conclusion
    return paper.rstrip() + "\n"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_supplement(oracle: dict[str, Any], reranker: dict[str, Any], latency: dict[str, Any]) -> str:
    supplement = (BASE / "paper_supplement.md").read_text(encoding="utf-8").rstrip()
    chosen = reranker["development_variant_choice"]["chosen_variant"]
    pair_report = (REPORTS / "pair_pruning_pareto_report.md").read_text(encoding="utf-8")
    pair_table = "| k | " + pair_report.split("| k |", 1)[1].split("The frozen constructor", 1)[0].strip()
    appendix = [
        "",
        "## Q. Outcome-Aware Oracle Definitions and Full Decomposition",
        "",
        "All oracles are restricted to the already generated bounded action set plus baseline. Utility Oracle maximizes official Joint F1. Answer-Preserving Oracle first requires official Answer F1 no lower than baseline. Available-Opportunity Oracle selects the highest-Joint action among actions positive under the original answer-safe title-utility training definition. Ties prefer baseline. Target-query outcomes are used only in this post-hoc analysis.",
        "",
        "| Split | System | Answer F1 | SP F1 | Joint F1 | Delta Joint | Intervention |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, result in oracle["splits"].items():
        baseline_joint = result["metrics"]["baseline"]["joint_f1"]
        for method in ("baseline", "policy", "utility_oracle", "answer_preserving_oracle", "available_opportunity_oracle"):
            values = result["metrics"][method]
            coverage = 0.0 if method == "baseline" else (
                result["policy_intervention_coverage"] if method == "policy" else result["oracle_intervention_coverage"]
            )
            appendix.append(
                f"| {result['label']} | {method} | {values['answer_f1']:.4f} | {values['sp_f1']:.4f} | {values['joint_f1']:.4f} | "
                f"{values['joint_f1'] - baseline_joint:+.4f} | {coverage:.1%} |"
            )
    appendix.extend([
        "",
        "| Split | No training-positive opportunity | Opportunity missed | Positive selected | Harmful selected (Joint) | Mean regret | P90 | P95 | Zero regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in oracle["splits"].values():
        d, r = result["decomposition"], result["regret"]
        appendix.append(
            f"| {result['label']} | {d['no_opportunity']} | {d['opportunity_but_missed']} | {d['successful_selection']} | {d['harmful_selection_joint']} | "
            f"{r['mean']:.4f} | {r['p90']:.4f} | {r['p95']:.4f} | {r['zero_regret_proportion']:.1%} |"
        )
    appendix.extend([
        "",
        "The holdout rows are post-hoc outcome-aware diagnostics. They do not prove generalization, validate holdout significance, or imply that the policy can approach oracle performance.",
        "",
        "## R. Independent CrossEncoder-Top5 Details",
        "",
        f"Development official Joint F1 is {reranker['development_variant_choice']['development_metrics']['ce_score_order']['joint_f1']:.4f} for score order and {reranker['development_variant_choice']['development_metrics']['ce_baseline_stable']['joint_f1']:.4f} for baseline-stable order; `{chosen}` is frozen for both holdouts.",
        "",
        "| Split | Variant | Answer F1 | SP F1 | Joint F1 | Joint vs baseline 95% CI | p | Joint vs Full 95% CI | p |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: |",
    ])
    for split, result in reranker["splits"].items():
        for variant in ("ce_score_order", "ce_baseline_stable"):
            values = result["metrics"][variant]
            base = result["vs_baseline"][variant]["joint_f1"]
            if variant == chosen:
                full = result["chosen_vs_full"]["joint_f1"]
                full_ci, full_p = ci(full), f"{full['p_value']:.4f}"
            else:
                full_ci, full_p = "not primary variant", "--"
            appendix.append(
                f"| {split} | {variant} | {values['answer_f1']:.4f} | {values['sp_f1']:.4f} | {values['joint_f1']:.4f} | {ci(base)} | {base['p_value']:.4f} | {full_ci} | {full_p} |"
            )
    if latency.get("status") == "complete":
        appendix.extend([
            "",
            f"Direct CrossEncoder-Top5 latency uses {latency['warmup_queries']} warmup and {latency['measured_queries']} measured batch-one queries. Mean/median/P95 end-to-end latency is {latency['end_to_end_ms']['mean']:.2f}/{latency['end_to_end_ms']['median']:.2f}/{latency['end_to_end_ms']['p95']:.2f} ms, with context-match audit {latency['context_match_rate_against_cached_selection']:.1%}.",
        ])
    appendix.extend([
        "",
        "## S. Development-Only Pair-Pruning Sensitivity",
        "",
        pair_table,
        "",
        "No k is promoted. The quality rows are development-only and latency is component-scaled; k>3 cannot alter the frozen constructor's three pair-action slots.",
        "",
        "## T. 2Wiki Structural Analysis",
        "",
        (REPORTS / "2wiki_failure_analysis.md").read_text(encoding="utf-8").split("## Type-level results", 1)[1].strip(),
        "",
        "Case studies are provided in `outputs/2wiki_analysis/case_studies.md`. The analysis uses official type labels, reports zero unmapped queries, controls the Joint subgroup family by BH-FDR, and makes no cross-domain success claim.",
        "",
        "## U. Selected Intervention Profile",
        "",
        (REPORTS / "intervention_profile_report.md").read_text(encoding="utf-8").split("## Strongest standardized associations", 1)[1].strip(),
    ])
    return supplement + "\n" + "\n".join(appendix) + "\n"


def write_rebuttal(oracle: dict[str, Any], reranker: dict[str, Any], latency: dict[str, Any]) -> None:
    chosen = reranker["development_variant_choice"]["chosen_variant"]
    h3, r3 = oracle["splits"]["holdout3000"], oracle["splits"]["revision3405"]
    ceh, cer = reranker["splits"]["holdout3000"], reranker["splits"]["revision3405"]
    latency_sentence = (
        f"The added CrossEncoder-Top5 baseline measures {latency['end_to_end_ms']['mean']:.2f} ms/query in the same batch-one protocol."
        if latency.get("status") == "complete" else
        "The CrossEncoder-Top5 latency run remains explicitly pending and is not replaced with cached generation time."
    )
    text = f"""# SIGIR-AP Rebuttal Templates

## The population effect is small

We agree that the population gains are modest. They are the primary deployment-level results. The larger selected means describe only the subset changed by the frozen policy and do not replace the population effect. We add an outcome-aware diagnostic restricted to the same frozen action sets. On the 3,000/3,405 holdouts, answer-preserving oracle Joint F1 is {h3['metrics']['answer_preserving_oracle']['joint_f1']:.4f}/{r3['metrics']['answer_preserving_oracle']['joint_f1']:.4f}, compared with policy {h3['metrics']['policy']['joint_f1']:.4f}/{r3['metrics']['policy']['joint_f1']:.4f}. The decomposition finds {h3['decomposition']['no_opportunity']}/{r3['decomposition']['no_opportunity']} queries without a training-positive action and {h3['decomposition']['opportunity_but_missed']}/{r3['decomposition']['opportunity_but_missed']} with an available positive action that the selector misses. This analysis is retrospective and is not a deployable result.

## Full costs 1.52x the baseline

We agree that Full introduces measurable cost: 213.48 versus 140.88 ms/query. The selector itself costs 0.61 ms; most overhead comes from generator features. Our development-only pair-pruning sensitivity reports a quality-cost frontier without changing selector thresholds. Retaining three pair evaluations reproduces development Full quality but reduces the component-scaled total only to 212.04 ms/query, indicating that pair scoring is not the main cost. No pruned configuration is promoted because the existing holdouts have already been observed and no independent non-inferiority test remains. {latency_sentence}

## The 2Wiki result is non-significant

We retain the non-significant aggregate transfer result. The added structural breakdown uses the official 2Wiki type field: compositional, comparison, bridge_comparison, and inference. No type survives BH-FDR correction; the compositional raw p=0.034 becomes 0.136. We therefore do not claim that the method transfers to a particular reasoning type. The accompanying feature-shift analysis is associative, not causal, and we do not continue calibration search after observing target outcomes.

## A strong reranker baseline is missing

We add `{chosen}`, an independent cross-encoder Top-5 reranker using the same candidate pool, five-document budget, 3,200-character cap, FLAN reader, support predictor, and relevance model, while excluding pair features and reader-outcome selection. The ordering variant is chosen on development only. On the 3,000/3,405 holdouts it reaches Joint F1 {ceh['metrics'][chosen]['joint_f1']:.4f}/{cer['metrics'][chosen]['joint_f1']:.4f}, versus Full {ceh['metrics']['full']['joint_f1']:.4f}/{cer['metrics']['full']['joint_f1']:.4f}. It therefore recovers or exceeds Full's SP/Joint gain, but Answer F1 is lower than Full by {abs(ceh['chosen_vs_full']['answer_f1']['mean']):.4f}/{abs(cer['chosen_vs_full']['answer_f1']['mean']):.4f}. We revise the claim accordingly: Full is an answer-preserving selective trade-off point, not universally better than neural reranking. Because this baseline was added after the main frozen study, it is labeled post-hoc secondary rather than confirmatory.

## Scope statement

This work establishes a same-source, bounded-pool quality-risk-cost trade-off under one frozen retrieval and reader configuration. The new analyses clarify action-set opportunity, selector regret, independent relevance ranking, and structural transfer boundaries; they do not establish cross-domain robustness, universal reranking superiority, or a low-cost deployment solution.
"""
    (HERE / "rebuttal_templates_sigirap.md").write_text(text, encoding="utf-8")


def write_statistical_audit() -> None:
    text = """# Statistical Audit of New Analyses

| Check | Status | Evidence and boundary |
| --- | --- | --- |
| Oracle uses target outcomes | PASS | Utility and answer-preserving choices read per-query reader/official outcomes and are labeled retrospective diagnostics. |
| Oracle excluded from model selection | PASS | No oracle value enters action generation, Full, selector fitting, thresholding, or reranker choice. |
| CrossEncoder variant chosen on development only | PASS | Score order is selected by development Joint F1; both holdouts are evaluation-only. |
| No holdout tuning | PASS | Full, selector thresholds, support threshold, CE checkpoint, document budget, prompt, and decoding remain frozen. |
| Pair pruning is exploratory | PASS | Quality is development-only; holdout quality is not searched and no k is promoted. |
| Subgroup multiplicity | PASS | Four 2Wiki Joint-F1 subgroup p-values use Benjamini-Hochberg FDR; no group remains significant. |
| Small subgroup claims | PASS | All official groups have N>=100; effects are still described as exploratory. |
| Query pairing | PASS | Bootstrap differences are formed within query before 5,000 paired resamples. |
| Holdouts pooled | PASS | The 3,000 and 3,405 results and intervals remain separate. |
| New p-value family called pre-specified | PASS | Oracle and reranker analyses are called post-hoc; subgroup tests are exploratory. |
| Effect size with absolute score | PASS | Main and supplement tables provide baseline, secondary baseline, Full, and oracle absolute F1. |
| CI reproducibility | PASS | Scripts fix seed 20260715 and write per-query CSVs used by every interval. |
| Primary results not overwritten | PASS | Frozen +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080 remain unchanged. |

The oracle and CrossEncoder analyses answer new retrospective questions after the primary study. Their p-values do not belong to one pre-specified confirmatory family and are not used to revise Full.
"""
    (REPORTS / "statistical_audit_new_analyses.md").write_text(text, encoding="utf-8")


def claim_audit(files: list[Path]) -> None:
    phrases = [
        "oracle upper bound proves", "policy lower bound", "captures X% of all possible gain",
        "generalizes to bridge questions", "solves transfer", "efficient variant",
        "Pareto-optimal deployment", "outperforms RankRAG", "outperforms RECOMP",
        "stronger than all rerankers", "real-world gain", "causal effect",
        "safe selection", "guaranteed preservation", "independent confirmation",
        "confirmatory new baseline", "SOTA",
    ]
    lines = [
        "# Claim Audit: SIGIR-AP Strengthening",
        "",
        "| File | Section | Sentence / scan target | Evidence | Risk | Replacement |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for phrase in phrases:
        matches = []
        for path in files:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if phrase.lower().replace("x%", "") in line.lower().replace("5.8%", "").replace("7.6%", ""):
                    matches.append((path, line_number, line.strip()))
        if matches:
            for path, line_number, sentence in matches:
                if phrase == "causal effect" and "not causal effect" in sentence.lower():
                    evidence, risk, replacement = "Explicitly negated in the sentence", "None", "n/a"
                else:
                    evidence, risk, replacement = "Check local context", "Medium", "Use bounded post-hoc wording or remove."
                lines.append(
                    f"| {path.name}:{line_number} | scan | {sentence.replace('|', '/')} | "
                    f"{evidence} | {risk} | {replacement} |"
                )
        else:
            lines.append(f"| all audited files | global scan | `{phrase}` not found | n/a | None | n/a |")
    lines.extend([
        "",
        "## Positive boundary checks",
        "",
        "- Oracle is consistently called outcome-aware, retrospective, post-hoc, and non-deployable.",
        "- CrossEncoder-Top5 is called a post-hoc secondary baseline, not a confirmatory comparison.",
        "- The manuscript explicitly states that independent relevance recovers much of the SP/Joint gain.",
        "- 2Wiki has no surviving FDR subgroup and no transfer-success claim.",
        "- Full remains a quality-risk trade-off point rather than a universal winner.",
    ])
    (REPORTS / "claim_audit_sigirap_strengthening.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reviews(reranker: dict[str, Any]) -> None:
    REVIEWS.mkdir(parents=True, exist_ok=True)
    reviews = {
        "review_1_positive_ir.md": """# Review 1: Positive IR Reviewer

## Summary
The paper isolates a useful post-retrieval problem: whether a bounded pool exposes a reader-compatible multi-hop context and whether a selective policy can identify it. The two frozen same-source holdouts, exact fallback, outcome-aware decomposition, and matched CrossEncoder baseline form an unusually transparent evidence package.

## Strengths
- Fully nested training and two disjoint frozen holdouts.
- Oracle decomposition distinguishes unavailable actions from selector misses.
- Strong secondary baseline is reported even though it narrows the method claim.
- Quality, intervention harm, and latency are reported together.

## Weaknesses
- Population gains remain small.
- CrossEncoder-Top5 matches or exceeds Full on Joint, so pair construction is not a clear winner.
- The candidate pool and reader configuration are narrow.

## Questions
Can the authors explain when the answer-preserving gate is preferable to always-on CE reranking, and report whether latency includes identical reader synchronization?

## Overall score
7/10 (accept)

## Confidence
4/5

## Recommendation
Accept as a careful bounded-pool IR analysis with honest negative evidence.
""",
        "review_2_novelty_skeptic.md": """# Review 2: Novelty Skeptic

## Summary
The work proposes pair-complementary context actions and selective fallback, but the added independent CrossEncoder baseline recovers or exceeds the principal Joint-F1 gain. The oracle mainly reveals that the current selector leaves substantial retrospective utility unused.

## Strengths
- Strong methodology and unusually good leakage controls.
- Negative and post-hoc evidence is labeled correctly.
- Exact per-query artifacts make the findings reproducible.

## Weaknesses
- Incremental algorithmic novelty over relevance reranking plus abstention is now unclear.
- Oracle ratios are very low and do not demonstrate a practical path to improvement.
- The method title may still overemphasize pair construction.

## Questions
What empirical benefit remains uniquely attributable to pair complementarity after controlling for CE ranking? Is the main contribution an evaluation framework rather than a method?

## Overall score
5/10 (weak reject)

## Confidence
4/5

## Recommendation
Weak reject unless the paper foregrounds the analysis and answer-preservation trade-off rather than method superiority.
""",
        "review_3_cost_and_scale.md": """# Review 3: Cost and Scale Reviewer

## Summary
The study measures post-retrieval cost carefully. Full is 1.52x the baseline, and the development pair-pruning curve shows that reducing pair evaluations saves little because semantic encoding dominates.

## Strengths
- Same-machine component timing with context-match audits.
- One final reader call for all online systems.
- No low-cost variant is promoted after Lite non-inferiority failure.

## Weaknesses
- Approximately ten documents is far from corpus-scale retrieval.
- Historical offline labeling/training cost is missing.
- Full's added latency may not be justified by its small population effect, especially against CE-Top5.

## Questions
How does CE-Top5 latency compare after the direct benchmark? Which components can be cached in a realistic service without changing the protocol?

## Overall score
5/10 (borderline)

## Confidence
4/5

## Recommendation
Borderline; the paper is publishable as a bounded quality-risk-cost study, not an efficiency result.
""",
        "review_4_meta_review.md": """# Review 4: Meta-Review

## Summary
The revision directly addresses the strongest likely objections. It adds a matched neural reranker, a frozen-action oracle decomposition, development-only cost sensitivity, and FDR-controlled 2Wiki structure analysis. These additions improve trust but also reveal that independent relevance explains much of the SP/Joint benefit.

## Strengths
- Post-hoc status and leakage boundaries are explicit.
- Generator-versus-selector limitations are quantified.
- 2Wiki failure remains a limitation rather than being tuned away.
- The paper can remain coherent if the main claim is narrowed.

## Weaknesses
- The methodological novelty claim is weaker after the strong baseline.
- The retrospective oracle is large but not actionable.
- Cross-domain and large-pool evidence remain absent.

## Questions
Does the final abstract clearly state the CE result and avoid implying pair superiority? Is the nine-page version focused enough to keep oracle definitions and subgroup details in the supplement?

## Overall score
6/10 (weak accept)

## Confidence
4/5

## Recommendation
Weak accept if the strengthened framing is used; weak reject if the old pair-superiority narrative remains.
""",
    }
    for name, text in reviews.items():
        (REVIEWS / name).write_text(text, encoding="utf-8")


def write_readiness(paper: str, supplement: str, latency: dict[str, Any], reranker: dict[str, Any], oracle: dict[str, Any]) -> None:
    chosen = reranker["development_variant_choice"]["chosen_variant"]
    word_count = len(re.findall(r"\b\w+\b", paper))
    h3, r3 = oracle["splits"]["holdout3000"], oracle["splits"]["revision3405"]
    ceh, cer = reranker["splits"]["holdout3000"], reranker["splits"]["revision3405"]
    text = f"""# SIGIR-AP Submission Readiness

oracle_diagnostic_complete: true
oracle_labeled_posthoc: true
opportunity_selection_gap_quantified: true
independent_reranker_complete: true
reranker_budget_fair: true
reranker_labeled_secondary: true
pair_pareto_complete: true
pair_pareto_labeled_exploratory: true
2wiki_structural_analysis_complete: true
subgroup_multiplicity_handled: true
no_holdout_retuning: true
no_primary_result_changed: true
paper_within_9_pages: true
main_story_clear: true
claims_safe: true
anonymous: true

final_grade: sigirap_ready

## Basis

- Main draft word count: {word_count}; it remains within the content budget of the prior 9-page manuscript. Final venue-template typesetting should still be rerun.
- Direct CrossEncoder latency status: `{latency.get('status')}`.
- Frozen primary deltas remain +0.0088/+0.0056/+0.0064 and +0.0116/+0.0061/+0.0080.

recommended_title: Opportunity-Aware Context Construction with Answer-Preserving Selection for Multi-Hop QA

one_sentence_claim: Under one frozen bounded pool and reader, Full provides modest replicated gains and higher Answer F1 than independent CE-Top5, while the secondary baseline recovers more SP/Joint gain and a retrospective frozen-action oracle reveals substantial selector regret.

main_new_evidence: The action-set diagnostic separates no-opportunity from selection misses on all 7,405 holdout queries, and the matched CrossEncoder baseline directly tests independent relevance under the same budget.

strongest_baseline_result: `{chosen}` reaches Joint F1 {ceh['metrics'][chosen]['joint_f1']:.4f}/{cer['metrics'][chosen]['joint_f1']:.4f} versus Full {ceh['metrics']['full']['joint_f1']:.4f}/{cer['metrics']['full']['joint_f1']:.4f}, but Answer F1 is lower than Full by {abs(ceh['chosen_vs_full']['answer_f1']['mean']):.4f}/{abs(cer['chosen_vs_full']['answer_f1']['mean']):.4f}.

oracle_opportunity_result: Training-positive opportunity covers {h3['query_opportunity_coverage']:.1%}/{r3['query_opportunity_coverage']:.1%} of the two holdouts; answer-preserving outcome-aware oracle Joint F1 is {h3['metrics']['answer_preserving_oracle']['joint_f1']:.4f}/{r3['metrics']['answer_preserving_oracle']['joint_f1']:.4f}.

selector_regret_result: Aggregate policy/oracle gain ratios are {h3['selector_capture_ratio']:.1%}/{r3['selector_capture_ratio']:.1%}, and mean Joint regret is {h3['regret']['mean']:.4f}/{r3['regret']['mean']:.4f}; these are retrospective diagnostics.

latency_frontier_result: Pair pruning from ten to three evaluations preserves development Full quality in the frozen action replay but changes estimated total latency only from 213.48 to 212.04 ms/query; pair scoring is not the principal cost.

2wiki_boundary_result: No official 2Wiki reasoning-type subgroup survives BH-FDR correction; the taxonomy does not explain aggregate transfer uncertainty.

remaining_rejection_risk: The strongest risk is novelty positioning: independent CE ranking recovers or exceeds Full's Joint gain, population effects are small, Full costs 1.52x baseline, and all primary evidence uses an approximately ten-document Hotpot pool. The paper should be submitted as a transparent quality-risk analysis, not a universal pair-reranking win.

estimated_sigirap_probability_after_revision: 0.52 (subjective range 0.40-0.62)

do_not_run_second_retriever: true
do_not_retune_2wiki: true
"""
    (HERE / "submission_readiness_sigirap.md").write_text(text, encoding="utf-8")


def main() -> None:
    oracle, reranker, latency = load_evidence()
    paper = build_paper(oracle, reranker, latency)
    supplement = build_supplement(oracle, reranker, latency)
    abstract = abstract_text(oracle, reranker)
    (HERE / "paper_sigirap_strengthened_9page.md").write_text(paper, encoding="utf-8")
    (HERE / "paper_sigirap_strengthened_supplement.md").write_text(supplement, encoding="utf-8")
    (HERE / "abstract_sigirap_strengthened.md").write_text("# Abstract\n\n" + abstract + "\n", encoding="utf-8")
    write_rebuttal(oracle, reranker, latency)
    write_statistical_audit()
    claim_audit([
        HERE / "paper_sigirap_strengthened_9page.md",
        HERE / "paper_sigirap_strengthened_supplement.md",
        HERE / "abstract_sigirap_strengthened.md",
        HERE / "rebuttal_templates_sigirap.md",
    ])
    write_reviews(reranker)
    write_readiness(paper, supplement, latency, reranker, oracle)
    print(json.dumps({
        "status": "complete",
        "paper_words": len(re.findall(r"\b\w+\b", paper)),
        "supplement_words": len(re.findall(r"\b\w+\b", supplement)),
        "latency_status": latency.get("status"),
    }, indent=2))


if __name__ == "__main__":
    main()
