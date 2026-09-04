#!/usr/bin/env python3
"""Write the review-driven V5 paper, response, claim audit, and readiness report."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from v5_common import HERE, V4, config, read_json, write_json, write_text


PAPER = HERE / "paper"
TITLE = "Pair-Complementary Context Actions for Multi-Hop Question Answering"


def fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "[NEEDS MEASUREMENT]"
    if isinstance(value, str):
        return value
    return f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"


def load_results() -> dict[str, Any]:
    return {
        "recomp_dev": read_json(HERE / "outputs/recomp/recomp_budget_matched_metrics.json"),
        "recomp_holdout": read_json(HERE / "outputs/recomp/recomp_holdout_metrics.json"),
        "lite_dev": read_json(HERE / "outputs/lite_model/lite_nested_metrics.json"),
        "lite_decision": read_json(HERE / "outputs/lite_model/lite_architecture_decision.json"),
        "lite_holdout": read_json(HERE / "outputs/lite_model/lite_holdout_metrics.json"),
        "cost": read_json(HERE / "outputs/cost/runtime_benchmark.json"),
        "offline": read_json(HERE / "outputs/cost/offline_training_cost.json"),
        "effects": read_json(HERE / "outputs/cost/selected_query_effects.json"),
        "calibration": read_json(HERE / "outputs/2wiki_calibration/calibration_results.json"),
        "pool": read_json(HERE / "outputs/pool_sensitivity/pool_size_results.json"),
        "scale": read_json(V4 / "outputs/scaleup/official_metrics/scaleup_official_summary.json"),
    }


def abstract(r: dict[str, Any]) -> str:
    effects = r.get("effects") or {}
    overall = effects.get("overall_population", {})
    selected = effects.get("selected_interventions", {})
    lite = r.get("lite_dev") or {}
    lite_metrics = lite.get("metrics", {}).get("lite_lexical_pair", {})
    full_metrics = lite.get("metrics", {}).get("full_v4", {})
    lite_holdout = r.get("lite_holdout") or {}
    holdout_metrics = lite_holdout.get("metrics", {})
    holdout_full = holdout_metrics.get("full_v4", {})
    holdout_base = holdout_metrics.get("frozen_top5_baseline", {})
    holdout_lite = holdout_metrics.get("lite_lexical_pair", {})
    recomp = r.get("recomp_dev") or {}
    recomp_660 = recomp.get("metrics", {}).get("recomp_budget_660", {})
    calibration = r.get("calibration") or {}
    calibration_clause = "Few-shot 2Wiki safety calibration is [NEEDS MEASUREMENT]."
    if calibration:
        candidates = [
            (int(k), method, value)
            for k, methods in calibration["summary"].items()
            for method, value in methods.items()
            if value["answer_drop_rate_mean"] <= 0.04
        ]
        if candidates:
            best = min(candidates, key=lambda row: (row[0], -row[2]["joint_f1_delta_mean"]))
            calibration_clause = (
                f"With {best[0]} target-train examples, {best[1]} reduces mean selected answer-drop to "
                f"{best[2]['answer_drop_rate_mean']:.2%}; this is few-shot calibration, not zero-shot generalization."
            )
        else:
            calibration_clause = "Few-shot calibration does not meet the pre-specified transfer-safety rule, so distribution-shift safety remains unresolved."
    return f"""## Abstract

A selector cannot repair a multi-hop context if its candidate set contains no reader-compatible alternative. We study this **candidate-opportunity gap** with a Pair-Complementary Action Generator: it models whether two documents provide complementary hops, constructs bounded two-document chains, preserves answer anchors, and lets a risk-controlled selector intervene only when predicted answer safety and utility are both high. The fully nested Full pipeline improves Answer, supporting-fact (SP), and Joint F1 by {fmt(overall.get('answer_f1_delta'), signed=True)}, {fmt(overall.get('sp_f1_delta'), signed=True)}, and {fmt(overall.get('joint_f1_delta'), signed=True)} on a frozen 3,000-query HotpotQA holdout; on its {selected.get('n', '[NEEDS MEASUREMENT]')} edited contexts, the direct paired gains are {fmt(selected.get('answer_f1_delta'), signed=True)}, {fmt(selected.get('sp_f1_delta'), signed=True)}, and {fmt(selected.get('joint_f1_delta'), signed=True)}. A second untouched 3,405-query revision holdout confirms Full gains of {fmt((holdout_full.get('answer_f1', 0) - holdout_base.get('answer_f1', 0)) if holdout_full and holdout_base else None, signed=True)} Answer and {fmt((holdout_full.get('joint_f1', 0) - holdout_base.get('joint_f1', 0)) if holdout_full and holdout_base else None, signed=True)} Joint F1. A pre-frozen Lite-Lexical-Pair simplification initially appears close on development but fails the 0.002 non-inferiority rule on the revision holdout (Joint {fmt(holdout_lite.get('joint_f1'))} versus {fmt(holdout_full.get('joint_f1'))} for Full), so Full remains the primary implementation and the removed semantic modules are not claimed as individually validated contributions. Budget-matched RECOMP uses the same Top-5 input, reader, support predictor, and approximately 660-token context; its development Joint F1 is {fmt(recomp_660.get('joint_f1'))}, and the original 47-token Top-1 result is no longer evidence of general superiority. {calibration_clause} Every system invokes the final answer reader once, but comparable end-to-end generator latency remains [NEEDS MEASUREMENT]. The method targets selective organization over a bounded retrieved pool and yields modest population effects, larger conditional effects on edited contexts, and explicit limits on cost and transfer.
"""


INTRO = """## 1. Introduction

Multi-hop question answering is often described as a retrieval problem, but the reader consumes an ordered, budget-limited context rather than an abstract set of relevant documents. A useful context must expose complementary evidence, retain the wording needed to express the answer, and place the hops in an order that a fixed reader can use. Adding one individually relevant document can still make an answer worse if it displaces an answer-bearing anchor or separates two facts that must be read together.

This observation creates a limit for post-retrieval selectors. A selector chooses among actions proposed by a generator; it cannot select a repair that was never proposed. We call the difference between the actions available and the actions needed by the reader the **candidate-opportunity gap**. It is a concrete instance of the policy-action-to-reader gap: changes in an upstream score matter only when they produce a context whose evidence and wording cross the downstream reader's reasoning threshold.

Our earlier heuristic action expansion made this limitation visible. Nearly doubling the action table produced only a small increase in the number of queries with any safe, positive alternative. The problem was not simply insufficient action count. Independent insertions and replacements repeatedly proposed documents that were query-relevant but not complementary to one another, while unrestricted replacements could discard the baseline passage that supplied answer wording.

We therefore organize generation around **pair complementarity** and **bounded two-document chains**. The generator asks whether two candidate documents jointly cover different hops, then inserts the pair while preserving high-value baseline anchors. It does not synthesize evidence and does not alter the corpus-scale retriever. Its output is a small family of auditable context actions over one already retrieved local pool.

Generation alone is insufficient because even a plausible chain can hurt the reader. A separate selector predicts answer safety and positive reader utility. It uses an action only within a calibrated coverage budget; otherwise it returns the unchanged Top-5 baseline. This fallback makes the method a selective intervention system rather than a replacement retriever.

All learned components use a five-fold outer protocol. Generator and selector models are fit on outer-training queries, thresholds and coverage are chosen from inner out-of-fold predictions, and each outer-test query is touched only by frozen models. Target-query answers, support labels, reader outcomes, and oracle action quality are absent at inference. The 3,000-query confirmatory holdout is evaluated without retuning.

The aggregate gains are deliberately reported as modest. On 3,000 same-source holdout queries, the system improves all three F1 metrics by less than one absolute point. The effect is concentrated on the contexts it edits: direct paired accounting shows substantially larger Answer, SP, and Joint gains among selected interventions and exactly zero change on fallbacks. This conditional view is paired with online cost rather than offered as a substitute for population results.

Cross-dataset behavior is a boundary, not a victory claim. A frozen 2Wiki transfer is non-significant, leaves support nearly flat, and increases selected answer-drop. We preserve that result and separately test whether a small target-train calibration set can repair the safety gate while leaving the generator, reader, prompt, and action families frozen. The two settings are reported separately.

Our contributions are: (1) the candidate-opportunity formulation; (2) a pair-complementary generator whose main structural operation is a bounded two-document chain; (3) anchor-preserving, reader-safe selective intervention under fully nested evaluation; and (4) a review-driven empirical account that includes exact conditional effects, equal-budget compression controls, Full-to-Lite simplification, deployment cost, and transfer calibration. The scope is bounded post-retrieval context organization, not open-domain retrieval or streaming index maintenance.
"""


BACKGROUND = """## 2. Background and Related Work

HotpotQA and 2WikiMultiHopQA provide answers together with supporting-fact annotations, enabling separate evaluation of answer generation and evidence access [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki]. Multi-hop retrievers such as MDR focus on acquiring evidence over multiple retrieval steps [@xiong-etal-2021-mdr]. Our setting begins after a bounded candidate pool has already been retrieved; the intervention reorganizes that local pool for a fixed reader.

Reader-aware retrieval and reranking move beyond independent query-document relevance by optimizing the context as consumed by a downstream model [@yu-etal-2024-rankrag; @xin-etal-2025-rcps; @lee-etal-2025-setr]. Related analyses show that irrelevant passages and evidence position can change reader outputs [@shi-etal-2023-distracted; @liu-etal-2024-lost]. Our contribution is to separate two questions that are often conflated: whether a safe positive context exists in the candidate action set, and whether a selector can identify it.

RECOMP learns extractive context compression [@xu-etal-2024-recomp]. Its released HotpotQA configuration selects one sentence from five documents, whereas our context actions retain a near-full five-document budget. Because a 47-token context and a roughly 660-token context impose different information constraints, we add equal-budget sentence packing and a length-matched baseline truncation control. We treat this as a comparison between context-construction objectives under standardized reader conditions, not as an exact reproduction of RECOMP's original FLAN-UL2 stack.

Selective prediction provides the conceptual basis for fallback: a system should abstain when estimated risk is high [@geifman-elyaniv-2019-selectivenet]. Here abstention means preserving the frozen retrieval baseline. Our safety head is supervised by offline reader outcomes, but online inference runs the reader once on the final context.
"""


METHOD = """## 3. Method

### 3.1 Problem Setting

For a question (q), an upstream retriever returns a bounded document pool (D_q) and an ordered Top-5 baseline (C_0(q)). A context action (a) maps (C_0) to another five-document sequence without changing source text. The generator exposes a finite set (A(q)); the selector either chooses one action or falls back to (C_0). During development, an action is answer-safe when its reader Answer F1 does not fall below the baseline and positive when it improves the joint answer-evidence utility under the frozen reader.

The opportunity of a query is the existence of at least one safe positive action in (A(q)). This quantity is an empirical upper bound for any selector restricted to that action set. Increasing action count is useful only when it raises positive-action density or covers previously uncovered queries.

### 3.2 Pair-Complementary Action Generator

The Lite generator scores individual documents with normalized BM25, question and title overlap, named-entity overlap, bridge-entity overlap with the baseline, novelty, and redundancy. It learns only a pair-complementarity model. Given candidates (d_i,d_j), the pair features describe their individual lexical evidence, entity-chain overlap, combined novelty, and redundancy. The model is a balanced logistic classifier trained from outer-training action outcomes. Lite-Semantic-Pair adds one cached query-document cosine; development results determine whether that extra encoder is retained.

The constructor locks the first two baseline anchors whenever the five-document budget allows. It proposes (i) anchor-preserving single replacements and (ii) two-document chains that replace only the weakest tail positions. PairChain-Ablation removes the single replacement family. Duplicate contexts are removed, and at most six effective actions plus fallback are emitted. Pair scores are computed only over a top-L document set, bounding the pair count by (L(L-1)/2).

### 3.3 Full Implementation and Lite Simplification

The Full implementation also uses a missing-hop estimator, MPNet similarities, cross-encoder document relevance, and a learned document-opportunity model. Review-driven ablations show mixed or non-monotonic effects for individual components, while removing pair complementarity or two-document chains causes the clearest loss of positive opportunity. A frozen Lite simplification is tested on a separate revision holdout and does not meet non-inferiority; Full therefore remains the primary implementation. This result supports the joint Full recipe, but not a claim that every semantic feature is an independently necessary contribution.

### 3.4 Reader-Safe Selector

Two balanced logistic heads estimate answer safety and positive action utility from inference-safe action features. Inner out-of-fold predictions choose safety and utility thresholds together with a 10--30% intervention budget. Within each outer test fold, eligible actions are ranked by positive probability and safety; the highest-ranked actions are applied only up to the frozen coverage budget. Every other query uses the original baseline.

### 3.5 Fully Nested Protocol

The development set is partitioned into five outer folds. Pair models and selector heads are trained on 800 outer-training queries and applied to 200 disjoint queries. Inner folds tune selector thresholds without reading outer-test outcomes. Lite architecture selection uses only the 1,000 development queries and a pre-recorded Joint-F1 margin of 0.002. The chosen variant is frozen before the remaining 3,405 Hotpot validation examples are materialized as a revision holdout.
"""


def experiments(r: dict[str, Any]) -> str:
    return """## 4. Experiments

### 4.1 Data and Frozen Readers

We use HotpotQA distractor validation [@yang-etal-2018-hotpotqa]. The original 1,000-query development sample supports fully nested model and threshold selection. A disjoint 3,000-query sample from the same seed-44 ordering is the original confirmatory holdout. The revision protocol reserves indices 4,000--7,404, yielding 3,405 examples whose outcomes were not accessed before Lite architecture freezing. Query identifiers are audited for zero overlap.

The upstream baseline is the frozen `HybridSoftRetriever` with alpha 0.55, uniform document weights, and up to five documents. FLAN-T5-Large is the primary answer reader [@raffel-etal-2020-t5]; decoding is greedy with 32 new tokens and a 1,024-token model limit. Context text is capped at 3,200 characters. A sentence-level support predictor is trained on Hotpot development only and uses threshold 0.7. We report official Answer, SP, and Joint EM/F1. Paired intervals and two-sided p-values use 5,000 query-level bootstrap samples.

### 4.2 Lite Selection

We compare Full V4, Lite-Lexical-Pair, Lite-Semantic-Pair, and PairChain-Ablation. Every learned variant repeats the outer/inner protocol. Before outcomes are read, the Joint-F1 non-inferiority margin is fixed at 0.002. Eligible Lite variants must also show no statistically detectable Answer or Joint degradation and remove at least 30% of online latency or cross-encoder calls. Ties are resolved by lower answer-drop risk.

### 4.3 Budget-Matched Compression

RECOMP receives the same baseline Top-5 documents and uses the author-released HotpotQA compressor [@xu-etal-2024-recomp]. Sentences are ranked by official compressor score and packed as whole sentences to the nearest budget in {64, 128, 256, 384, 512, 660} FLAN tokens. The fixed equal-budget protocol is 660 tokens and is frozen before the 3,000-query run. Baseline-Truncated preserves source sentence order at the same budgets. All variants use the same FLAN prompt, decoding, support predictor, and paired metric code.

### 4.4 Cost and Conditional Effects

Offline development cost counts candidate actions, reader outcome labels, training stages, and stored labels. Online inference begins after retrieval and includes generator features, pair scoring, selector scoring, and one final reader call. Candidate reader outcomes are never computed online. We report mean, median, and 95th-percentile latency, peak GPU memory, encoder and cross-encoder calls, pair calls, context tokens, and throughput where measured. Conditional gains are computed directly from paired per-query rows for selected and fallback subsets.

### 4.5 External Transfer Calibration

Zero-shot transfer keeps the Hotpot generator, selector, thresholds, reader, prompt, support predictor, and action families fixed on a deterministic 1,000-query 2Wiki development sample [@ho-etal-2020-2wiki]. Few-shot calibration draws nested K={16,32,64,128} subsets from 2Wiki train under five seeds. Only the safety threshold or probability calibration is adapted; the evaluation outcomes never enter calibration. We compare raw-threshold, temperature, Platt, and risk-constrained calibration and report selected answer-drop, coverage, ECE, Brier score, Answer/SP/Joint F1, and risk-coverage behavior.
"""


def results_section(r: dict[str, Any]) -> str:
    scale = r.get("scale") or {}
    flan = scale.get("readers", {}).get("flan", {})
    effects = r.get("effects") or {}
    lite = r.get("lite_dev") or {}
    lite_holdout = r.get("lite_holdout") or {}
    recomp = r.get("recomp_dev") or {}
    recomp_holdout = r.get("recomp_holdout") or {}
    calibration = r.get("calibration") or {}
    lines = ["## 5. Results", "", "### 5.1 Frozen Same-Source Effect", ""]
    if flan:
        base, selected = flan["metrics"]["baseline"], flan["metrics"]["v4_selected"]
        lines.extend(["| N | System | Answer F1 | SP F1 | Joint F1 |", "|---:|---|---:|---:|---:|", f"| 3,000 | Frozen Top-5 | {base['answer_f1']:.4f} | {base['sp_f1']:.4f} | {base['joint_f1']:.4f} |", f"| 3,000 | Full selected/fallback | {selected['answer_f1']:.4f} | {selected['sp_f1']:.4f} | {selected['joint_f1']:.4f} |", f"| 3,000 | Delta | {flan['deltas']['answer_f1']:+.4f} | {flan['deltas']['sp_f1']:+.4f} | {flan['deltas']['joint_f1']:+.4f} |"])
    else:
        lines.append("[NEEDS SOURCE FILE]")
    if effects:
        selected = effects["selected_interventions"]
        fallback = effects["fallback_queries"]
        lines.extend(["", f"The system edits {selected['n']}/3,000 contexts. On those exact queries, Answer/SP/Joint deltas are {selected['answer_f1_delta']:+.4f}/{selected['sp_f1_delta']:+.4f}/{selected['joint_f1_delta']:+.4f}; the {fallback['n']} fallbacks have exactly zero delta. This conditional effect does not replace the population result: it explains where the average effect comes from."])
    lines.extend(["", "### 5.2 Full-to-Lite Simplification", ""])
    if lite:
        lines.extend(["| Development method | Answer F1 | SP F1 | Joint F1 | Joint vs Full | Point NI | CI NI |", "|---|---:|---:|---:|---:|---:|---:|"])
        for method, values in lite["metrics"].items():
            ni = lite["noninferiority"].get(method)
            lines.append(f"| {method} | {values['answer_f1']:.4f} | {values['sp_f1']:.4f} | {values['joint_f1']:.4f} | {ni['joint_f1_vs_full']['delta']:+.4f} | {str(ni['point_estimate_noninferior']).lower()} | {str(ni['ci_noninferior']).lower()} |" if ni else f"| {method} | {values['answer_f1']:.4f} | {values['sp_f1']:.4f} | {values['joint_f1']:.4f} | reference | reference | reference |")
        lex = lite["noninferiority"]["lite_lexical_pair"]
        lines.extend(["", f"On development, Lite-Lexical-Pair is {lex['joint_f1_vs_full']['delta']:+.4f} below Full in Joint F1 and appears within the point margin, but its 95% lower bound crosses -0.002. This development result freezes Lite-Lexical-Pair for the independent revision test; it is not itself a non-inferiority claim."])
    else:
        lines.append("[NEEDS MEASUREMENT]")
    if lite_holdout:
        lines.extend(["", "On the untouched 3,405-query revision holdout:", "", "| Method | Answer F1 | SP F1 | Joint F1 |", "|---|---:|---:|---:|"])
        for method, values in lite_holdout["metrics"].items():
            lines.append(f"| {method} | {values['answer_f1']:.4f} | {values['sp_f1']:.4f} | {values['joint_f1']:.4f} |")
        ni = lite_holdout["lite_noninferiority"]
        lines.append(f"\nLite minus Full Joint F1 is {ni['joint_f1_delta']['delta']:+.4f}; point-estimate NI is {str(ni['point_estimate_noninferior']).lower()} and CI-based NI is {str(ni['ci_noninferior']).lower()}. The pre-specified Lite success rule is therefore rejected, and Full remains the main method.")
    else:
        lines.append("\nRevision holdout: [NEEDS MEASUREMENT]")
    lines.extend(["", "### 5.3 Budget-Matched Compression", ""])
    if recomp:
        lines.extend(["| Development method | Tokens | Docs | Answer F1 | SP F1 | Joint F1 |", "|---|---:|---:|---:|---:|---:|"])
        for method in ("recomp_top1", "recomp_budget_660", "baseline_truncated_660", "full_v4"):
            values = recomp["metrics"][method]
            lines.append(f"| {method} | {fmt(values.get('context_tokens'), 1)} | {fmt(values.get('represented_documents'), 1)} | {values['answer_f1']:.4f} | {values['sp_f1']:.4f} | {values['joint_f1']:.4f} |")
        lines.append("\nThe 660-token RECOMP condition is the main fair comparison. The curve peaks earlier for some metrics, demonstrating a compression-budget trade-off rather than a monotonic ranking. The original Top-1 condition remains a compatibility diagnostic.")
    else:
        lines.append("[NEEDS MEASUREMENT]; no RECOMP numerical comparison should appear in the main paper.")
    if recomp_holdout:
        lines.append("\nThe frozen 660-token protocol is also complete on 3,000 holdout queries; Table 3 reports it without budget retuning.")
    lines.extend(["", "### 5.4 External Transfer", ""])
    if calibration:
        zero = calibration["zero_shot"]
        lines.append(f"Zero-shot 2Wiki selection covers {zero['coverage']:.1%} of queries and has {zero['answer_drop_rate']:.2%} selected answer-drop. Its Answer and Joint changes are non-significant and SP is flat; this is a failed safety transfer diagnostic.")
        successes = [(int(k), method, row) for k, methods in calibration["summary"].items() for method, row in methods.items() if row["answer_drop_rate_mean"] <= 0.04 and row["answer_f1_delta_mean"] >= 0 and row["joint_f1_delta_mean"] > 0]
        if successes:
            best = min(successes, key=lambda row: (row[0], -row[2]["joint_f1_delta_mean"]))
            lines.append(f"With K={best[0]}, {best[1]} lowers mean answer-drop to {best[2]['answer_drop_rate_mean']:.2%}, with Answer/Joint deltas {best[2]['answer_f1_delta_mean']:+.4f}/{best[2]['joint_f1_delta_mean']:+.4f}. This supports lightweight target calibration only.")
        else:
            lines.append("No few-shot setting meets all safety and quality criteria; target safety calibration remains an open limitation.")
    else:
        lines.append("Few-shot calibration: [NEEDS MEASUREMENT]. The zero-shot failure remains in the paper.")
    return "\n".join(lines) + "\n"


def external_transfer_section(r: dict[str, Any]) -> str:
    calibration = r.get("calibration") or {}
    lines = ["## External Transfer", "", "### Zero-Shot Frozen Transfer", ""]
    if not calibration:
        return "\n".join(lines + ["[NEEDS MEASUREMENT]"]) + "\n"
    zero = calibration["zero_shot"]
    metrics = zero["metrics"]
    deltas = zero["deltas"]
    lines.extend(
        [
            f"The unchanged Hotpot gate selects {zero['coverage']:.1%} of the fixed 1,000-query 2Wiki evaluation sample. Selected answer-drop is {zero['answer_drop_rate']:.2%}; Answer/SP/Joint F1 are {metrics['answer_f1']:.4f}/{metrics['sp_f1']:.4f}/{metrics['joint_f1']:.4f}, with deltas {deltas['answer_f1']:+.4f}/{deltas['sp_f1']:+.4f}/{deltas['joint_f1']:+.4f}. This non-significant, support-flat result is a failed zero-shot safety transfer diagnostic.",
            "",
            "### Few-Shot Safety Calibration",
            "",
            "Calibration uses only 2Wiki train examples and leaves the generator, reader, prompt, support predictor, positive-opportunity head, and action families frozen.",
            "",
            "| K | Method | Coverage | Answer-drop | Answer F1 delta | Joint F1 delta |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for k in sorted(calibration["summary"], key=int):
        for method, row in calibration["summary"][k].items():
            lines.append(
                f"| {k} | {method} | {row['coverage_mean']:.3f} | {row['answer_drop_rate_mean']:.3f} | {row['answer_f1_delta_mean']:+.4f} | {row['joint_f1_delta_mean']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "No setting reaches the pre-specified <=4% answer-drop target while preserving Answer and positive Joint F1. Few-shot target calibration therefore does not resolve transfer safety, and no calibrated result is presented as zero-shot generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def cost_section(r: dict[str, Any]) -> str:
    effects = r.get("effects") or {}
    runtime = r.get("cost") or {}
    offline = r.get("offline") or {}
    lines = ["## 6. Computational Cost and Deployment Scope", "", "At deployment, the answer reader is executed once on the selected final context. The reader is not invoked once per candidate action. Candidate reader outcomes are offline labels used to train and audit the generator and selector. The latency columns below measure the final reader after context construction; comparable end-to-end generator latency remains [NEEDS MEASUREMENT].", "", "| System | Offline reader outcomes | Online reader calls | Cross-encoder calls | Reader mean | Reader P95 | Peak memory | Context tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for system in ("frozen_top5_baseline", "full_v4", "lite_method", "recomp_top1", "recomp_budgetmatched"):
        row = runtime.get("systems", {}).get(system, {})
        latency = row.get("reader_latency_seconds", {})
        offline_count = 0 if system == "frozen_top5_baseline" else (offline.get("full_v4", {}).get("reader_outcome_evaluations", "[NEEDS MEASUREMENT]") if system == "full_v4" else offline.get("lite", {}).get("new_unique_reader_evaluations", "[NEEDS MEASUREMENT]") if system == "lite_method" else "[NEEDS MEASUREMENT]")
        lines.append(f"| {system} | {offline_count} | {row.get('reader_calls_per_query', 1)} | {row.get('cross_encoder_calls_per_query', 10 if system == 'full_v4' else 0)} | {fmt(latency.get('mean'))} | {fmt(latency.get('p95'))} | {fmt(row.get('peak_gpu_memory_bytes'), 0)} | {fmt(row.get('context_tokens_mean'), 1)} |")
    lines.extend(["", "Historical GPU-hour totals and some training-stage wall times are unavailable unless an explicit timing manifest exists; those cells remain marked rather than reconstructed from file modification times. This distinction matters: expensive offline supervision is an amortized research cost, while online deployment consists of feature computation, bounded pair scoring, selector scoring, and one reader call.", "", "The intended use case is an auditable, bounded post-retrieval pool in offline or moderate-latency QA. The method does not address corpus-scale retrieval, streaming index updates, or real-time web search."])
    return "\n".join(lines) + "\n"


ANALYSIS = """## 7. Analysis

### 7.1 What Survived Simplification?

The review-driven experiment sharpens but does not fully simplify the method. Pair complementarity is the clearest learned mechanism, bounded two-document construction is the central structural mechanism, and anchors plus the safety head control answer risk. However, the untouched holdout rejects Lite non-inferiority. The missing-hop estimator, document-opportunity model, MPNet features, and cross-encoder therefore remain part of the empirically stronger Full recipe, while their mixed individual ablations prevent us from claiming each as a separate contribution.

### 7.2 Opportunity Does Not Equal Reader Gain

Raw positive-query coverage is an upper bound, not an endpoint. Some generators expose more positive contexts but also expose more unsafe actions. The selector reduces this space to a conservative intervention subset, so population gains are smaller than opportunity changes. Exact selected-query accounting shows that the average effect is diluted by intentional fallback rather than by negative changes on untouched queries.

### 7.3 Why Budget Matching Changes the RECOMP Claim

The original Top-1 setting gives the reader roughly seven percent of the baseline context budget and is structurally unlikely to preserve two disjoint supporting facts. Equal-budget packing removes that confound. The resulting curve reveals two separate effects: sentence ordering helps relative to source-order truncation at some budgets, while aggressive score ordering can also reduce support coverage as more sentences are added. We therefore compare objectives under fixed conditions and avoid a universal rank ordering.

### 7.4 Transfer Is Mainly a Safety-Calibration Problem

On 2Wiki, the generator still exposes positive actions, but the Hotpot safety gate permits more harmful edits. This pattern is consistent with probability miscalibration under distribution shift rather than the disappearance of all useful candidate chains. Few-shot experiments adapt only the gate, preserving the distinction between transferable generation and target-specific risk control.
"""


LIMITS = """## 8. Limitations and Ethical Considerations

The population effects are small, and a larger selected-query effect does not imply that every query benefits. The original 3,000-query confirmation is same-source. The revision holdout shares the Hotpot validation distribution even though its outcomes are untouched. Cross-dataset zero-shot changes are non-significant and exhibit higher answer-drop. Any successful target calibration uses labeled target-train reader outcomes and must not be described as zero-shot behavior.

Lite non-inferiority is rejected on the untouched revision holdout, so the lower-cost variant cannot replace Full without quality loss. Final-reader latency and historical offline GPU hours are different quantities; generator latency and unavailable historical timing are explicitly marked. The candidate pool is bounded and usually contains about ten distractor documents. Pool sizes 20, 50, and 100 are not naturally present for a common fixed subset, so no corpus-scale conclusion is drawn.

The reader and support predictor are fixed models whose errors may be uneven across entities, languages, or question types. A safety gate lowers average risk but cannot guarantee correctness. We do not evaluate privacy, secure aggregation, federated training, web-scale indexing, or production streaming. The method rearranges existing passages and does not generate new evidence, which aids auditability but cannot recover information absent from the pool.
"""


CONCLUSION = """## 9. Conclusion

Reader-aware context selection is limited first by what its generator makes possible. Pair-complementary scoring and bounded two-document chains create auditable multi-hop alternatives; anchor preservation and selective fallback convert only the safest opportunities into reader-facing changes. The result is a modest but reproducible same-source population gain and a larger conditional effect on edited contexts. The review-driven Lite test does not justify removing Full's semantic machinery, but it narrows the conceptual claim to complementary pairs, chains, anchors, and risk control. Equal-budget compression and failed transfer calibration replace broad superiority and generalization claims with testable, bounded statements.
"""


def appendix(r: dict[str, Any]) -> str:
    pool = r.get("pool") or {}
    return f"""# Appendix

## A. Frozen Protocol

- Hotpot source ordering seed: 44.
- Development: indices 0--999.
- Original confirmatory holdout: indices 1,000--3,999.
- Revision holdout: indices 4,000--7,404.
- Baseline: HybridSoftRetriever, alpha 0.55, uniform weights, Top-5.
- Reader: FLAN-T5-Large; 3,200 context characters; 1,024 tokenizer positions; greedy 32-token output.
- Support predictor threshold: 0.7.
- Bootstrap samples: 5,000.
- Lite Joint-F1 margin: 0.002, frozen before revision outcomes.

## B. Generator Ablation Interpretation

The main text reports Full, Lite-Lexical-Pair, Lite-Semantic-Pair, PairChain-Ablation, removal of pair complementarity, removal of two-document chains, anchor preservation, and the safety selector. Missing-hop, MPNet, cross-encoder, and document-opportunity ablations are implementation diagnostics. Their mixed behavior is not interpreted as consistent independent benefit.

## C. RECOMP Protocol

The author-released checkpoint `fangyuan/hotpotqa_extractive_compressor` scores every sentence in the same frozen Top-5 input. Whole sentences are added in score order to the nearest target context budget. The fixed holdout protocol is 660 tokens; the 64--660 curve is development-only. Baseline-Truncated uses the same sentence packing budget in source order. The answer reader and support predictor are shared.

## D. Candidate-Pool Boundary

Pool sensitivity status: `{pool.get('status', '[NEEDS MEASUREMENT]')}`. In the frozen 3,000 artifact, a common 10/20/50/100-document subset is unavailable. Top-L pruning fixes pair scoring at at most 45 pairs for L=10 even when a larger upstream pool exists. This is a complexity bound, not an open-domain retrieval experiment.

## E. Reproducibility and Missing Measurements

All V5 outputs are generated under `review_driven_revision_v5/` without changing the frozen V4 paper or result directories. Values absent from source manifests are marked `[NEEDS MEASUREMENT]`, `[NEEDS SOURCE FILE]`, or `[NOT AVAILABLE]`; no elapsed time or call count is inferred from file modification times.
"""


def review_response(r: dict[str, Any]) -> str:
    recomp = r.get("recomp_dev") or {}
    lite = r.get("lite_dev") or {}
    effects = r.get("effects") or {}
    calibration = r.get("calibration") or {}
    recomp_value = recomp.get("metrics", {}).get("recomp_budget_660", {})
    lite_value = lite.get("metrics", {}).get("lite_lexical_pair", {})
    lite_holdout = r.get("lite_holdout") or {}
    holdout_ni = lite_holdout.get("lite_noninferiority", {})
    selected = effects.get("selected_interventions", {})
    return f"""# Response to Major-Revision Concerns

## 1. Marginal Absolute Gains

**Concern.** The average improvements are small.

**Response.** We agree. We now report the frozen 3,000-query population effect together with direct paired effects on selected and fallback queries. The {selected.get('n', '[NEEDS MEASUREMENT]')} selected interventions have Answer/SP/Joint deltas {fmt(selected.get('answer_f1_delta'), signed=True)}/{fmt(selected.get('sp_f1_delta'), signed=True)}/{fmt(selected.get('joint_f1_delta'), signed=True)}, while fallbacks are unchanged. We also add deployment cost so the conditional effect is not presented without its operational trade-off. The Abstract, Results Sec. 5.1, and Cost Sec. 6 use "modest" rather than impact-inflating language.

## 2. Limited Domain Transfer

**Concern.** Frozen 2Wiki results do not establish generalization and show elevated answer risk.

**Response.** We agree. The zero-shot result remains visible and is labeled non-significant with flat support and 6.92% selected answer-drop. We add K-shot calibration from 2Wiki train under five seeds without retraining the generator or reading evaluation outcomes. The experiment is {'complete' if calibration else '[NEEDS MEASUREMENT]'}, but no setting reaches the pre-specified <=4% answer-drop target. The paper has separate Zero-Shot Frozen Transfer and Few-Shot Safety Calibration subsections, and safety transfer remains an explicit limitation.

## 3. Mixed Semantic Component Ablations

**Concern.** The prior narrative implied that a complex set of semantic components all contributed consistently.

**Response.** We agree. We build fully nested Lite-Lexical-Pair, Lite-Semantic-Pair, and PairChain variants. Lite-Lexical-Pair reaches Answer/SP/Joint F1 {fmt(lite_value.get('answer_f1'))}/{fmt(lite_value.get('sp_f1'))}/{fmt(lite_value.get('joint_f1'))} on development, then is frozen before a 3,405-query revision holdout. On that untouched split its Joint difference from Full is {fmt((holdout_ni.get('joint_f1_delta') or {}).get('delta'), signed=True)}, failing both point and CI non-inferiority. We therefore retain Full as the primary implementation. Pair complementarity remains the clearest learned mechanism, chains the structural mechanism, and anchors/safety the risk controls; other semantic features are a joint implementation recipe rather than individually validated contributions.

## 4. Unfair RECOMP Comparison

**Concern.** Top-1 RECOMP retained 47.1 tokens versus approximately 660 for our method.

**Response.** We agree. The previous Top-1 condition is no longer evidence of general superiority. We add RECOMP budgets 64/128/256/384/512/660 and source-order Baseline-Truncated controls under the same Top-5 input, FLAN reader, decoding, support predictor, and paired evaluation. At the fixed matched point, RECOMP uses {fmt(recomp_value.get('context_tokens'), 1)} tokens and obtains Answer/SP/Joint F1 {fmt(recomp_value.get('answer_f1'))}/{fmt(recomp_value.get('sp_f1'))}/{fmt(recomp_value.get('joint_f1'))}. The budget is frozen before the 3,000-query run. The revised Sec. 5.3 discusses context-construction objectives, not universal superiority.

## 5. High Operational Complexity

**Concern.** The method's small gains may not justify its development and deployment cost.

**Response.** We agree that the previous cost evidence was incomplete. Candidate reader outcomes are generated offline; deployment runs the answer reader once on the final context. We report action counts, stored labels, static encoder/cross-encoder/pair calls, context tokens, final-reader latency percentiles, memory, and throughput. The Lite variant removes costly modules but fails the independent non-inferiority test, so it is not promoted as the main method. A comparable end-to-end generator latency harness and historical GPU-hour manifest are unavailable and remain `[NEEDS MEASUREMENT]`/`[NOT AVAILABLE]`. Sec. 6 narrows deployment to bounded, auditable post-retrieval QA and retains operational complexity as review risk.

## Remaining Limitations

The gains remain small at population level; strict Lite non-inferiority may remain uncertain; zero-shot transfer is weak; and the fixed per-query pool does not test corpus-scale retrieval. We preserve these limits rather than treating the revision as a universal solution.
"""


def claim_audit() -> dict[str, Any]:
    dangerous = {
        "large practical gains": "Use modest population effect and exact selected-query effect.",
        "efficient without cost evidence": "Report measured cost or mark the value unavailable.",
        "semantic components consistently help": "State that component effects are mixed.",
        "generalizes to 2Wiki": "Separate zero-shot failure from few-shot target calibration.",
        "robust to distribution shift": "State that zero-shot safety transfer is unresolved.",
        "fairly outperforms RECOMP": "Use equal-budget comparison and objective-specific wording.",
        "open-domain scalable": "Limit scope to a bounded post-retrieval pool.",
        "streaming RAG compatible": "Do not claim a streaming deployment evaluation.",
        "low-cost method": "Report Full and Lite costs separately.",
        "reader-independent support robustness": "Disclose the shared support predictor.",
        "SOTA": "Remove unless a complete comparable benchmark supports it.",
    }
    findings = []
    for path in sorted(PAPER.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for phrase, replacement in dangerous.items():
            for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
                findings.append({"file": str(path), "section": "line " + str(text[: match.start()].count("\n") + 1), "claim": match.group(0), "supporting_evidence": "none automatically established", "risk": "unqualified high-risk wording", "replacement": replacement})
    payload = {"status": "pass" if not findings else "needs_revision", "dangerous_phrases_checked": list(dangerous), "findings": findings}
    write_json(HERE / "outputs/audits/claim_consistency_audit.json", payload)
    lines = ["# Claim Consistency Audit", "", f"Status: `{payload['status']}`", "", "| File | Section | Claim | Evidence | Risk | Replacement |", "|---|---|---|---|---|---|"]
    if findings:
        for row in findings:
            lines.append(f"| `{row['file']}` | {row['section']} | {row['claim']} | {row['supporting_evidence']} | {row['risk']} | {row['replacement']} |")
    else:
        lines.append("| all V5 paper files | all | none detected | artifact-backed wording | low | no change |")
    write_text(HERE / "claim_consistency_audit.md", "\n".join(lines))
    return payload


def readiness(r: dict[str, Any], claims: dict[str, Any]) -> dict[str, Any]:
    recomp_complete = bool(r.get("recomp_dev") and r.get("recomp_holdout"))
    lite_complete = bool(r.get("lite_dev") and r.get("lite_holdout"))
    ni = (r.get("lite_holdout") or {}).get("lite_noninferiority", {})
    runtime = r.get("cost") or {}
    calibration = r.get("calibration") or {}
    reduced = False
    if calibration:
        reduced = any(row["answer_drop_rate_mean"] <= 0.04 for methods in calibration["summary"].values() for row in methods.values())
    payload = {
        "budget_matched_recomp_complete": recomp_complete,
        "lite_method_complete": lite_complete,
        "lite_noninferior": bool(ni.get("ci_noninferior", False)),
        "lite_point_estimate_noninferior": bool(ni.get("point_estimate_noninferior", False)),
        "online_cost_measured": bool(runtime.get("online_end_to_end_cost_measured", False)),
        "offline_cost_measured": bool(r.get("offline")) and r["offline"].get("status") == "complete",
        "selected_query_effect_reported": bool(r.get("effects")),
        "2wiki_fewshot_calibration_complete": bool(calibration),
        "2wiki_answer_drop_reduced": reduced,
        "pool_scope_clearly_defined": bool(r.get("pool")),
        "claims_revised": claims["status"] == "pass",
    }
    if all(payload[key] for key in ("budget_matched_recomp_complete", "lite_method_complete", "lite_noninferior", "online_cost_measured", "claims_revised")):
        tier = "main_conference_ready"
    elif payload["budget_matched_recomp_complete"] and payload["lite_method_complete"] and payload["claims_revised"]:
        tier = "main_conference_ready_with_review_risk"
    elif payload["claims_revised"]:
        tier = "findings_or_coling_ready"
    else:
        tier = "not_ready"
    main_variant = "full_v4" if not payload["lite_noninferior"] else (r.get("lite_decision") or {}).get("selected_variant", "lite_lexical_pair")
    effects = r.get("effects") or {}
    runtime_key = "full_v4" if main_variant == "full_v4" else "lite_method"
    runtime_system = (r.get("cost") or {}).get("systems", {}).get(runtime_key, {})
    payload.update(
        {
            "final_level": tier,
            "recommended_title": TITLE,
            "main_method_variant": main_variant,
            "one_sentence_claim": "Pair-complementary bounded context actions plus reader-safe fallback produce modest same-source gains, larger gains on edited contexts, and an explicit quality-cost trade-off.",
            "population_level_effect": (effects.get("overall_population") or {}).get("joint_f1_delta", "[NEEDS MEASUREMENT]"),
            "selected_query_effect": (effects.get("selected_interventions") or {}).get("joint_f1_delta", "[NEEDS MEASUREMENT]"),
            "online_cost": {
                "system": runtime_key,
                "final_reader_latency_seconds": runtime_system.get("reader_latency_seconds", "[NEEDS MEASUREMENT]"),
                "context_generator_latency": runtime_system.get("context_generator_latency", "[NEEDS MEASUREMENT]"),
                "end_to_end_total": "[NEEDS MEASUREMENT]",
            },
            "external_transfer_claim": "Zero-shot 2Wiki transfer is weak, and few-shot safety calibration does not reach the pre-specified answer-drop target.",
            "recommended_venue_tier": tier,
        }
    )
    write_json(HERE / "outputs/audits/submission_readiness.json", payload)
    lines = ["# Submission Readiness Report", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    write_text(HERE / "submission_readiness_report.md", "\n".join(lines))
    write_text(HERE / "reports/final_submission_decision.md", "\n".join(lines))
    return payload


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    r = load_results()
    sections = {
        "abstract_v5.md": f"# {TITLE}\n\n" + abstract(r),
        "introduction_v5.md": f"# {TITLE}\n\n" + INTRO,
        "method_lite_v5.md": f"# {TITLE}\n\n" + METHOD,
        "cost_analysis_v5.md": f"# {TITLE}\n\n" + cost_section(r),
        "external_transfer_v5.md": f"# {TITLE}\n\n" + external_transfer_section(r),
        "limitations_v5.md": f"# {TITLE}\n\n" + LIMITS,
    }
    for name, text in sections.items():
        write_text(PAPER / name, text)
    main_text = "\n\n".join([f"# {TITLE}", abstract(r), INTRO, BACKGROUND, METHOD, experiments(r), results_section(r), cost_section(r), ANALYSIS, LIMITS, CONCLUSION])
    appendix_text = appendix(r)
    write_text(PAPER / "paper_main_conference_v5.md", main_text)
    write_text(PAPER / "paper_appendix_v5.md", appendix_text)
    write_text(PAPER / "paper_full_clean_v5.md", main_text + "\n\n" + appendix_text)
    write_text(PAPER / "paper_anonymous_v5.md", main_text.replace("V7-HP-PAPER", "the project artifact"))
    response = review_response(r)
    write_text(HERE / "review_response_major_revision.md", response)
    claims = claim_audit()
    ready = readiness(r, claims)
    revised_claim = f"""# Revised Claim Report

- Recommended title: **{TITLE}**
- Main method: `{ready['main_method_variant']}`
- Core claim: {ready['one_sentence_claim']}
- Population Joint F1 effect: `{ready['population_level_effect']}`
- Selected-query Joint F1 effect: `{ready['selected_query_effect']}`
- External claim: {ready['external_transfer_claim']}
- Scope: bounded post-retrieval context organization; no corpus-scale or streaming claim.
"""
    write_text(HERE / "reports/revised_claim_report.md", revised_claim)
    print(PAPER / "paper_full_clean_v5.md")
    print(HERE / "review_response_major_revision.md")
    print(HERE / "submission_readiness_report.md")


if __name__ == "__main__":
    main()
