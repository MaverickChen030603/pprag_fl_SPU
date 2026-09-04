#!/usr/bin/env python3
"""Exploratory 2Wiki structural and failure-attribution analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import joblib

from sigirap_common import (
    COMPLETION,
    FIGURES,
    OUTPUTS,
    REPORTS,
    V4,
    ensure_layout,
    iter_jsonl,
    paired_bootstrap,
    read_json,
    write_json,
)


EXTERNAL = COMPLETION / "outputs/external_2wiki_frozen"
OUT = OUTPUTS / "2wiki_analysis"
METRICS = ("answer_f1", "sp_f1", "joint_f1")
EPS = 1e-12


def bh_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    previous = 1.0
    count = len(ordered)
    for reverse_rank, key in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        value = min(previous, p_values[key] * count / rank)
        adjusted[key] = min(1.0, value)
        previous = value
    return adjusted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def context_tokens(action: dict[str, Any]) -> int:
    text = "\n".join(f"{doc['title']}: {doc['text']}" for doc in action["context_docs"])
    return len(text[:3200].split())


def action_positive(outcome: dict[str, Any], baseline: dict[str, Any]) -> bool:
    answer_delta = float(outcome["answer_f1"]) - float(baseline["answer_f1"])
    recall_delta = float(outcome["title_recall"]) - float(baseline["title_recall"])
    title_f1_delta = float(outcome["title_f1"]) - float(baseline["title_f1"])
    product_delta = float(outcome["answer_title_product"]) - float(baseline["answer_title_product"])
    return answer_delta >= -EPS and product_delta > EPS and (recall_delta > EPS or title_f1_delta >= -EPS)


def cache_features(cache: dict[str, Any], query_id: str) -> dict[str, float]:
    query = cache["queries"][query_id]
    details = list(query["doc_feature_details"].values())
    pair_index = cache["pair_feature_names"].index("semantic_complementarity")
    pair_values = [float(values[pair_index]) for values in query["pair_features"].values()]
    doc_lengths = [len(str(doc["text"]).split()) for doc in query["docs"]]
    return {
        "question_length": len(str(query["question"]).split()),
        "document_length": mean(doc_lengths),
        "candidate_pool_size": len(query["docs"]),
        "entity_overlap": mean(float(value["entity_overlap"]) for value in details),
        "bridge_entity_frequency": mean(float(value["bridge_entity_match"]) > 0 for value in details),
        "pair_complementarity": mean(pair_values) if pair_values else 0.0,
    }


def main() -> None:
    ensure_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    source = {str(row["query_id"]): row for row in read_json(EXTERNAL / "2wiki_frozen_1000.json")}
    type_counts = Counter(str(row.get("type", "unmapped")) for row in source.values())
    if type_counts.get("unmapped", 0):
        grouping_status = "official type field with unmapped values retained"
    else:
        grouping_status = "official 2Wiki type field; no heuristic mapping"

    official_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(EXTERNAL / "official_per_query.jsonl"):
        official_rows[str(row["query_id"])][str(row["method"])] = row
    selections = {str(row["query_id"]): row for row in iter_jsonl(EXTERNAL / "frozen_selector_selections_1000.jsonl")}
    actions = {str(row["action_id"]): row for row in iter_jsonl(EXTERNAL / "generated_actions_1000.jsonl")}
    outcomes = {str(row["action_id"]): row for row in iter_jsonl(EXTERNAL / "reader/all_action_outcomes.jsonl")}
    cache = joblib.load(EXTERNAL / "semantic_feature_cache_1000.joblib")

    grouped_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions.values():
        grouped_actions[str(action["query_id"])].append(action)

    query_rows: list[dict[str, Any]] = []
    for query_id, item in source.items():
        baseline_id = f"{query_id}::v4xfer::fallback"
        selection = selections[query_id]
        selected_id = str(selection["action_id"])
        baseline_official = official_rows[query_id]["baseline"]
        full_official = official_rows[query_id]["v4_frozen_transfer"]
        baseline_outcome = outcomes[baseline_id]
        positives = [
            action for action in grouped_actions[query_id]
            if action["action_family"] != "fallback" and action_positive(outcomes[str(action["action_id"])], baseline_outcome)
        ]
        features = cache_features(cache, query_id)
        selected_action = actions[selected_id]
        query_rows.append({
            "query_id": query_id,
            "reasoning_type": str(item.get("type", "unmapped")),
            "selected": bool(selection["selected"]),
            "fallback": bool(selection["fallback"]),
            "available_opportunity": bool(positives),
            "answer_drop": bool(selection["selected"] and full_official["answer_f1"] < baseline_official["answer_f1"] - EPS),
            "joint_drop": bool(selection["selected"] and full_official["joint_f1"] < baseline_official["joint_f1"] - EPS),
            "context_tokens": context_tokens(selected_action),
            "pred_preservation": float(selection.get("pred_answer_safe_prob", 0.0)),
            "pred_utility": float(selection.get("pred_positive_prob", 0.0)),
            "action_family": str(selection["action_family"]),
            **features,
            **{f"baseline_{metric}": float(baseline_official[metric]) for metric in METRICS},
            **{f"full_{metric}": float(full_official[metric]) for metric in METRICS},
            **{f"delta_{metric}": float(full_official[metric]) - float(baseline_official[metric]) for metric in METRICS},
        })

    type_rows: list[dict[str, Any]] = []
    raw_joint_p: dict[str, float] = {}
    for reasoning_type in sorted(type_counts):
        rows = [row for row in query_rows if row["reasoning_type"] == reasoning_type]
        statistics = {metric: paired_bootstrap([row[f"delta_{metric}"] for row in rows]) for metric in METRICS}
        raw_joint_p[reasoning_type] = statistics["joint_f1"]["p_value"]
        type_rows.append({
            "reasoning_type": reasoning_type,
            "n": len(rows),
            **{f"baseline_{metric}": mean(row[f"baseline_{metric}"] for row in rows) for metric in METRICS},
            **{f"delta_{metric}": mean(row[f"delta_{metric}"] for row in rows) for metric in METRICS},
            **{f"{metric}_ci_low": statistics[metric]["ci95_low"] for metric in METRICS},
            **{f"{metric}_ci_high": statistics[metric]["ci95_high"] for metric in METRICS},
            **{f"{metric}_p": statistics[metric]["p_value"] for metric in METRICS},
            "candidate_opportunity_coverage": mean(row["available_opportunity"] for row in rows),
            "policy_coverage": mean(row["selected"] for row in rows),
            "answer_drop_rate_population": mean(row["answer_drop"] for row in rows),
            "joint_drop_rate_population": mean(row["joint_drop"] for row in rows),
            "average_candidate_documents": mean(row["candidate_pool_size"] for row in rows),
            "average_context_tokens": mean(row["context_tokens"] for row in rows),
            "average_entity_overlap": mean(row["entity_overlap"] for row in rows),
            "average_pair_complementarity": mean(row["pair_complementarity"] for row in rows),
            "fallback_rate": mean(row["fallback"] for row in rows),
            "small_group_descriptive_only": len(rows) < 50,
        })
    adjusted = bh_adjust(raw_joint_p)
    for row in type_rows:
        row["joint_f1_p_bh"] = adjusted[row["reasoning_type"]]
        row["joint_f1_fdr_significant"] = len([x for x in query_rows if x["reasoning_type"] == row["reasoning_type"]]) >= 50 and adjusted[row["reasoning_type"]] < 0.05
    write_csv(OUT / "type_metrics.csv", type_rows)

    # Same-source Hotpot holdout supplies a stable comparison distribution.
    hotpot_cache = joblib.load(V4 / "outputs/scaleup/semantic_feature_cache_3000.joblib")
    hotpot_selections = {str(row["query_id"]): row for row in iter_jsonl(V4 / "outputs/scaleup/frozen_selector_selections_3000.jsonl")}
    hotpot_official: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(V4 / "outputs/scaleup/official_metrics/flan_per_query.jsonl"):
        hotpot_official[str(row["query_id"])][str(row["method"])] = row
    hotpot_rows = []
    for query_id in hotpot_selections:
        feature = cache_features(hotpot_cache, query_id)
        selection = hotpot_selections[query_id]
        metrics = hotpot_official[query_id]
        hotpot_rows.append({
            **feature,
            "pred_preservation": float(selection.get("pred_answer_safe_prob", 0.0)),
            "pred_utility": float(selection.get("pred_positive_prob", 0.0)),
            "selected": bool(selection["selected"]),
            "actual_answer_non_degrade": metrics["v4_selected"]["answer_f1"] >= metrics["baseline"]["answer_f1"] - EPS,
            "actual_joint_gain": metrics["v4_selected"]["joint_f1"] > metrics["baseline"]["joint_f1"] + EPS,
            "action_family": str(selection["action_family"]),
        })
    wiki_rows = [{
        **{key: row[key] for key in ("question_length", "document_length", "candidate_pool_size", "entity_overlap", "bridge_entity_frequency", "pair_complementarity", "pred_preservation", "pred_utility", "selected", "action_family")},
        "actual_answer_non_degrade": row["full_answer_f1"] >= row["baseline_answer_f1"] - EPS,
        "actual_joint_gain": row["full_joint_f1"] > row["baseline_joint_f1"] + EPS,
    } for row in query_rows]

    distribution_rows: list[dict[str, Any]] = []
    for feature in ("question_length", "document_length", "candidate_pool_size", "entity_overlap", "bridge_entity_frequency", "pair_complementarity", "pred_preservation", "pred_utility"):
        hotpot_values = [float(row[feature]) for row in hotpot_rows]
        wiki_values = [float(row[feature]) for row in wiki_rows]
        try:
            from scipy.stats import ks_2samp
            test = ks_2samp(hotpot_values, wiki_values)
            ks_stat, ks_p = float(test.statistic), float(test.pvalue)
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")
        distribution_rows.append({
            "feature": feature,
            "hotpot_mean": mean(hotpot_values),
            "2wiki_mean": mean(wiki_values),
            "mean_shift_2wiki_minus_hotpot": mean(wiki_values) - mean(hotpot_values),
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
        })
    for family in sorted(set(row["action_family"] for row in hotpot_rows + wiki_rows)):
        distribution_rows.append({
            "feature": f"action_family_frequency::{family}",
            "hotpot_mean": mean(row["action_family"] == family for row in hotpot_rows),
            "2wiki_mean": mean(row["action_family"] == family for row in wiki_rows),
            "mean_shift_2wiki_minus_hotpot": mean(row["action_family"] == family for row in wiki_rows) - mean(row["action_family"] == family for row in hotpot_rows),
            "ks_statistic": "",
            "ks_p_value": "",
        })
    write_csv(OUT / "distribution_shift.csv", distribution_rows)

    # Calibration and risk-coverage are descriptive because fallback rows carry the frozen zero score.
    calibration: dict[str, list[dict[str, float]]] = {}
    risk_curves: dict[str, list[dict[str, float]]] = {}
    for reasoning_type in sorted(type_counts):
        rows = [row for row in query_rows if row["reasoning_type"] == reasoning_type]
        bins = []
        for index in range(5):
            low, high = index / 5, (index + 1) / 5
            members = [row for row in rows if low <= row["pred_preservation"] < high or (index == 4 and row["pred_preservation"] == 1.0)]
            if members:
                bins.append({
                    "bin_low": low,
                    "bin_high": high,
                    "n": len(members),
                    "predicted": mean(row["pred_preservation"] for row in members),
                    "observed_non_degrade": mean(row["full_answer_f1"] >= row["baseline_answer_f1"] - EPS for row in members),
                })
        calibration[reasoning_type] = bins
        scored = sorted(rows, key=lambda row: row["pred_preservation"] * row["pred_utility"], reverse=True)
        curve = []
        for coverage in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            retained = scored[: max(1, math.ceil(len(scored) * coverage))]
            curve.append({
                "coverage": coverage,
                "answer_drop_rate": mean(row["full_answer_f1"] < row["baseline_answer_f1"] - EPS for row in retained),
                "joint_delta": mean(row["delta_joint_f1"] for row in retained),
            })
        risk_curves[reasoning_type] = curve
    write_json(OUT / "calibration_and_risk_curves.json", {"calibration": calibration, "risk_coverage": risk_curves})

    # Cases are selected by deterministic effect ordering, never by a model-training decision.
    successes = sorted(
        [row for row in query_rows if row["selected"] and row["delta_joint_f1"] > EPS],
        key=lambda row: row["delta_joint_f1"], reverse=True,
    )[:3]
    harmful = sorted(
        [row for row in query_rows if row["selected"] and row["delta_joint_f1"] < -EPS],
        key=lambda row: row["delta_joint_f1"],
    )[:3]
    missed = [row for row in query_rows if row["fallback"] and row["available_opportunity"]][:2]
    case_lines = ["# 2Wiki Structural Case Studies", "", "These are post-hoc explanatory cases; no case was used to tune the frozen system.", ""]
    for label, cases in (("Successful transfer", successes), ("Harmful selected action", harmful), ("Fallback despite available opportunity", missed)):
        case_lines.extend([f"## {label}", ""])
        for row in cases:
            query_id = row["query_id"]
            item = source[query_id]
            baseline_id = f"{query_id}::v4xfer::fallback"
            selection = selections[query_id]
            selected_id = str(selection["action_id"])
            shown_id = selected_id
            if label.startswith("Fallback"):
                candidates = [
                    action for action in grouped_actions[query_id]
                    if action["action_family"] != "fallback" and action_positive(outcomes[str(action["action_id"])], outcomes[baseline_id])
                ]
                shown_id = max(candidates, key=lambda action: outcomes[str(action["action_id"])]["answer_title_product"])["action_id"]
            shown = actions[str(shown_id)]
            baseline = actions[baseline_id]
            case_lines.extend([
                f"### {query_id}",
                "",
                f"- Question: {item['question']}",
                f"- Reasoning type: `{item.get('type', 'unmapped')}`.",
                f"- Relevant entity chain (gold, analysis only): {' -> '.join(item['supporting_titles'])}.",
                f"- Baseline titles: {', '.join(baseline['context_titles'])}.",
                f"- Shown action `{shown['action_family']}`: {', '.join(shown['context_titles'])}.",
                f"- Reader answer before / after: `{outcomes[baseline_id]['prediction']}` / `{outcomes[str(shown_id)]['prediction']}`.",
                f"- Pair-complementarity reading: the action {'adds or reorders a missing supporting title' if set(map(str.lower, item['supporting_titles'])) - set(map(str.lower, baseline['context_titles'])) else 'reorders evidence already present'}; the observed Joint-F1 change is {row['delta_joint_f1']:+.4f} for the frozen selected policy.",
                "",
            ])
    (OUT / "case_studies.md").write_text("\n".join(case_lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    names = [row["reasoning_type"] for row in type_rows]
    values = [row["delta_joint_f1"] for row in type_rows]
    low = [row["delta_joint_f1"] - row["joint_f1_ci_low"] for row in type_rows]
    high = [row["joint_f1_ci_high"] - row["delta_joint_f1"] for row in type_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.errorbar(names, values, yerr=[low, high], fmt="o", color="#235789", capsize=4)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_ylabel("Full minus baseline Joint F1")
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "2wiki_type_effects.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for reasoning_type, curve in risk_curves.items():
        ax.plot([row["coverage"] for row in curve], [row["answer_drop_rate"] for row in curve], marker="o", label=reasoning_type)
    ax.set_xlabel("Descriptive score coverage")
    ax.set_ylabel("Answer-drop rate")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "2wiki_type_risk_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# 2Wiki Reasoning-Type Failure Analysis",
        "",
        "## Taxonomy audit",
        "",
        f"Grouping uses the dataset's actual `type` field: **{grouping_status}**. Counts are " + ", ".join(f"{key}={value}" for key, value in sorted(type_counts.items())) + ". The unmapped proportion is " + f"{type_counts.get('unmapped', 0)/len(source):.1%}.",
        "",
        "## Type-level results",
        "",
        "| Type | N | Baseline A/SP/J | Full delta A/SP/J | Joint 95% CI | raw p | BH-FDR p | opportunity | policy coverage |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in type_rows:
        report.append(
            f"| {row['reasoning_type']} | {row['n']} | {row['baseline_answer_f1']:.4f}/{row['baseline_sp_f1']:.4f}/{row['baseline_joint_f1']:.4f} | "
            f"{row['delta_answer_f1']:+.4f}/{row['delta_sp_f1']:+.4f}/{row['delta_joint_f1']:+.4f} | "
            f"[{row['joint_f1_ci_low']:+.4f}, {row['joint_f1_ci_high']:+.4f}] | {row['joint_f1_p']:.4f} | {row['joint_f1_p_bh']:.4f} | "
            f"{row['candidate_opportunity_coverage']:.1%} | {row['policy_coverage']:.1%} |"
        )
    significant = [row["reasoning_type"] for row in type_rows if row["joint_f1_fdr_significant"]]
    report.extend([
        "",
        "## Interpretation",
        "",
        ("After BH-FDR correction, the following sufficiently sized type groups retain a non-zero Joint effect: " + ", ".join(significant) + ". This remains descriptive heterogeneity, not a universal generalization claim." if significant else "No reasoning-type subgroup retains a statistically resolved Joint effect after BH-FDR correction. The available taxonomy therefore does not by itself explain the aggregate transfer uncertainty."),
        "",
        "The distribution-shift table compares Hotpot-3,000 and 2Wiki on question/document length, pool size, entity overlap, bridge frequency, pair complementarity, policy scores, and action-family frequencies. Associations with transfer behavior are descriptive and do not establish causality.",
    ])
    (REPORTS / "2wiki_failure_analysis.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "type_counts": type_counts, "fdr_significant": significant}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
