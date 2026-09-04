#!/usr/bin/env python3
"""Development-only pair-scoring cost/quality sensitivity analysis."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from statistics import mean
from typing import Any

import joblib

from sigirap_common import FIGURES, OUTPUTS, REPORTS, SPLITS, V4, ensure_layout, iter_jsonl, load_module


OUT = OUTPUTS / "pareto"
K_VALUES = (1, 2, 3, 5, 7, 10)
PAIR_FEATURE_MS = 1.7732006444130093
PAIR_SCORE_MS = 0.29196500312536955
FULL_TOTAL_MS = 213.48419773590285
FULL_GENERATOR_MS = 70.04637512832414
EPS = 1e-12


def pair_rank(action: dict[str, Any]) -> int | None:
    if action.get("action_family") != "semantic_two_document_chain":
        return None
    name = str(action.get("action_name", ""))
    if name in {"semantic_pair_1", "semantic_pair_bridge_middle"}:
        return 1
    if name == "semantic_pair_2":
        return 2
    if name == "semantic_pair_3":
        return 3
    return 10


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    official_path = OUTPUTS / "oracle/official_all_actions_development1000.jsonl"
    if not official_path.exists():
        raise FileNotFoundError(
            f"{official_path} is required; complete the outcome-aware diagnostic first"
        )
    if str(V4) not in sys.path:
        sys.path.insert(0, str(V4))
    selector = load_module(V4 / "07_train_nested_selector_v4.py", "sigirap_pair_selector")
    actions = list(iter_jsonl(SPLITS["development1000"]["actions"]))
    official = {str(row["action_id"]): row for row in iter_jsonl(official_path)}
    outcome_labels = {
        str(row["action_id"]): bool(row.get("positive_action", False))
        for row in iter_jsonl(V4 / "outputs/action_outcomes/v4_action_outputs.jsonl")
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[str(action["query_id"])].append(action)
    all_ids = set(grouped)

    results: list[dict[str, Any]] = []
    for k in K_VALUES:
        selected_by_query: dict[str, dict[str, Any]] = {}
        retained_by_query: dict[str, list[dict[str, Any]]] = {}
        for query_id, values in grouped.items():
            retained_by_query[query_id] = [
                row for row in values
                if row["action_family"] == "fallback" or pair_rank(row) is None or pair_rank(row) <= k
            ]
        for outer_fold in range(5):
            test_ids = {
                query_id for query_id, values in retained_by_query.items()
                if int(values[0]["outer_fold"]) == outer_fold
            }
            effective = [
                row for query_id in test_ids for row in retained_by_query[query_id]
                if row["action_family"] != "fallback"
            ]
            bundle = joblib.load(V4 / f"outputs/scaleup/selector_models/fold_{outer_fold}_selector.joblib")
            safe = selector.probabilities(bundle["safety_model"], effective)
            positive = selector.probabilities(bundle["opportunity_model"], effective)
            scored = []
            for row, safe_probability, positive_probability in zip(effective, safe, positive):
                value = dict(row)
                value["pred_answer_safe_prob"] = float(safe_probability)
                value["pred_positive_prob"] = float(positive_probability)
                scored.append(value)
            config = bundle["config"]
            selected_by_query.update(selector.select(
                scored,
                test_ids,
                float(config["safe_threshold"]),
                float(config["positive_threshold"]),
                float(config["coverage"]),
            ))

        metric_rows = []
        opportunity_queries = 0
        positive_actions = 0
        effective_actions = 0
        answer_drops = 0
        for query_id in sorted(all_ids):
            fallback = next(row for row in retained_by_query[query_id] if row["action_family"] == "fallback")
            effective = [row for row in retained_by_query[query_id] if row["action_family"] != "fallback"]
            positives = [row for row in effective if outcome_labels[str(row["action_id"])]]
            opportunity_queries += int(bool(positives))
            positive_actions += len(positives)
            effective_actions += len(effective)
            current = selected_by_query.get(query_id, fallback)
            current_metrics = official[str(current["action_id"])]
            baseline_metrics = official[str(fallback["action_id"])]
            answer_drops += int(
                str(current["action_id"]) != str(fallback["action_id"])
                and float(current_metrics["answer_f1"]) < float(baseline_metrics["answer_f1"]) - EPS
            )
            metric_rows.append({
                "answer_f1": float(current_metrics["answer_f1"]),
                "sp_f1": float(current_metrics["sp_f1"]),
                "joint_f1": float(current_metrics["joint_f1"]),
                "baseline_joint_f1": float(baseline_metrics["joint_f1"]),
                "selected": str(current["action_id"]) != str(fallback["action_id"]),
            })
        pair_fraction = k / 10
        generator_ms = FULL_GENERATOR_MS - PAIR_FEATURE_MS - PAIR_SCORE_MS + pair_fraction * (PAIR_FEATURE_MS + PAIR_SCORE_MS)
        total_ms = FULL_TOTAL_MS - PAIR_FEATURE_MS - PAIR_SCORE_MS + pair_fraction * (PAIR_FEATURE_MS + PAIR_SCORE_MS)
        results.append({
            "k": k,
            "n_queries": len(metric_rows),
            "candidate_opportunity_coverage": opportunity_queries / len(metric_rows),
            "positive_action_density": positive_actions / max(1, effective_actions),
            "policy_coverage": mean(row["selected"] for row in metric_rows),
            "answer_f1": mean(row["answer_f1"] for row in metric_rows),
            "sp_f1": mean(row["sp_f1"] for row in metric_rows),
            "joint_f1": mean(row["joint_f1"] for row in metric_rows),
            "joint_f1_delta": mean(row["joint_f1"] - row["baseline_joint_f1"] for row in metric_rows),
            "answer_drop_population_rate": answer_drops / len(metric_rows),
            "generator_latency_ms_estimated": generator_ms,
            "total_latency_ms_estimated": total_ms,
            "cross_encoder_calls": 10,
            "pairs_scored": k,
            "pair_feature_memory_relative_to_k10": pair_fraction,
            "selector_threshold_changed": False,
            "development_only": True,
        })
    write_csv(OUT / "pair_pruning_results.csv", results)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot([row["total_latency_ms_estimated"] for row in results], [row["joint_f1_delta"] for row in results], marker="o", color="#235789")
    for row in results:
        ax.annotate(f"k={row['k']}", (row["total_latency_ms_estimated"], row["joint_f1_delta"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Estimated total latency (ms/query)")
    ax.set_ylabel("Development Joint F1 delta")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "quality_latency_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot([row["generator_latency_ms_estimated"] for row in results], [row["candidate_opportunity_coverage"] for row in results], marker="o", color="#2a7f62")
    for row in results:
        ax.annotate(f"k={row['k']}", (row["generator_latency_ms_estimated"], row["candidate_opportunity_coverage"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Estimated generator latency (ms/query)")
    ax.set_ylabel("Positive-action opportunity coverage")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "opportunity_generator_cost.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Development-Only Pair-Pruning Pareto Analysis",
        "",
        "This optional analysis changes only the number of retained pair evaluations. Document features, frozen selector models, selector thresholds/coverage, reader outcomes, and action families are unchanged. Quality is evaluated only on the fully nested 1,000-query development outputs. Latency is an auditable component-scaled estimate from the frozen same-machine benchmark: pair-feature construction and pair scoring are scaled by k/10; all other measured components are held fixed.",
        "",
        "| k | Opportunity | Positive density | Policy coverage | Answer F1 | SP F1 | Joint F1 | Joint delta | Answer-drop | Generator ms | Total ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        report.append(
            f"| {row['k']} | {row['candidate_opportunity_coverage']:.1%} | {row['positive_action_density']:.1%} | "
            f"{row['policy_coverage']:.1%} | {row['answer_f1']:.4f} | {row['sp_f1']:.4f} | {row['joint_f1']:.4f} | "
            f"{row['joint_f1_delta']:+.4f} | {row['answer_drop_population_rate']:.1%} | "
            f"{row['generator_latency_ms_estimated']:.2f} | {row['total_latency_ms_estimated']:.2f} |"
        )
    report.extend([
        "",
        "The frozen constructor emits at most three ranked pair-chain action slots, so k>3 can reduce neither action diversity nor quality in this replay; those rows only expose the measured pair-scoring cost slope. This is an exploratory sensitivity analysis. No k is promoted as a new primary method because both existing holdouts have already been observed and no independent non-inferiority test remains.",
    ])
    (REPORTS / "pair_pruning_pareto_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "development_only": True, "rows": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
