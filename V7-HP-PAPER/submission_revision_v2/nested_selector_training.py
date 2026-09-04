#!/usr/bin/env python3
"""Run the fully nested V7-HP selector and submission diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from nested_safe_answer_feature_generation import (
    ROOT,
    V23,
    build_outer_fold_features,
    query_fingerprint,
    write_json,
    write_jsonl,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ACTIONS = ROOT / "V7-HP-PAPER" / "selector_v2_3" / "outputs" / "labels" / "action_labels.jsonl"
DEFAULT_OUTPUT = HERE / "outputs" / "nested"

PRIMARY_PROTOCOL = {
    "name": "nested_lexicographic_two_stage",
    "model_types": ["two_stage"],
    "objective": "answer-safety gate, then answer-safe joint-positive utility, else fallback",
    "utility_weights_used_for_primary": False,
    "config_search_scope": "outer_train_only",
    "nuisance_feature": "inner-query-OOF on outer train; full outer-train model on outer test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--skip-weight-sensitivity", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def config_from_best(best: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model_type",
        "selected_fraction",
        "answer_safe_threshold",
        "positive_threshold",
        "answer_drop_threshold",
        "candidate_family",
        "answer_margin",
    ]
    return {key: best[key] for key in keys}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate_selected(
    rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return V23.summarize_selection(selected, V23.group_by_query(rows))


def run_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outer_folds = V23.split_queries(rows, 5)
    all_selected: dict[str, dict[str, Any]] = {}
    fold_records = []
    feature_audits = []
    fold_runtime = []

    for fold_id, (outer_train_ids, outer_test_ids) in enumerate(outer_folds):
        train_rows, test_rows, feature_audit = build_outer_fold_features(
            rows, outer_train_ids, outer_test_ids, fold_id
        )
        models = V23.train_models(train_rows)
        best, top = V23.choose_config(
            train_rows,
            models,
            model_types=PRIMARY_PROTOCOL["model_types"],
            compact=True,
        )
        config = config_from_best(best)
        selected = V23.select_actions(test_rows, models, config)
        heldout_summary, heldout_per = V23.summarize_selection(selected, V23.group_by_query(test_rows))
        all_selected.update(selected)
        feature_audits.append(feature_audit)
        fold_records.append(
            {
                "fold_id": fold_id,
                "n_train": len(outer_train_ids),
                "n_test": len(outer_test_ids),
                "train_query_fingerprint": query_fingerprint(outer_train_ids),
                "test_query_fingerprint": query_fingerprint(outer_test_ids),
                "config": config,
                "train_best": best,
                "heldout_summary": heldout_summary,
                "top_train_configs": top[:10],
            }
        )
        fold_runtime.append(
            {
                "fold_id": fold_id,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "models": models,
                "config": config,
                "heldout_per": heldout_per,
            }
        )

    summary, per = aggregate_selected(rows, all_selected)
    summary["protocol"] = PRIMARY_PROTOCOL
    summary["fold_config_distribution"] = dict(
        Counter(json.dumps(record["config"], sort_keys=True) for record in fold_records)
    )
    return {
        "summary": summary,
        "per": per,
        "fold_records": fold_records,
        "feature_audits": feature_audits,
        "fold_runtime": fold_runtime,
    }


def run_nested_variant(
    rows: list[dict[str, Any]],
    name: str,
    model_types: list[str],
    drop_support: bool = False,
    drop_nested_safety: bool = False,
    compact: bool = True,
    target_weights: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    all_selected: dict[str, dict[str, Any]] = {}
    fold_configs = []
    for fold_id, (outer_train_ids, outer_test_ids) in enumerate(V23.split_queries(rows, 5)):
        train_rows, test_rows, _ = build_outer_fold_features(rows, outer_train_ids, outer_test_ids, fold_id)
        if target_weights is not None:
            answer_w, support_w, title_w = target_weights
            for row in train_rows + test_rows:
                row["listwise_target_score"] = (
                    V23.fnum(row, "joint_f1_delta")
                    + answer_w * V23.fnum(row, "answer_f1_delta")
                    + support_w * V23.fnum(row, "support_recall_delta")
                    + title_w * V23.fnum(row, "sp_f1_delta")
                )
        if drop_nested_safety:
            for row in train_rows + test_rows:
                row["safe_answer_prob"] = 0.5
                row["nested_safe_answer_prob"] = None
                row["safe_answer_prob_source"] = "removed_ablation"
        models = V23.train_models(
            train_rows,
            drop_support=drop_support,
            drop_safety=drop_nested_safety,
        )
        best, _ = V23.choose_config(
            train_rows,
            models,
            model_types=model_types,
            compact=compact,
        )
        config = config_from_best(best)
        selected = V23.select_actions(test_rows, models, config)
        heldout, _ = V23.summarize_selection(selected, V23.group_by_query(test_rows))
        all_selected.update(selected)
        fold_configs.append({"fold_id": fold_id, "config": config, "heldout": heldout})
    summary, _ = aggregate_selected(rows, all_selected)
    summary["ablation_name"] = name
    summary["fold_configs"] = fold_configs
    if target_weights is not None:
        summary["utility_weights"] = {
            "answer": target_weights[0],
            "support": target_weights[1],
            "title_f1": target_weights[2],
            "product": 1.0,
        }
    return summary


def risk_coverage(
    rows: list[dict[str, Any]],
    fold_runtime: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    output_rows = []
    per_by_coverage: dict[str, list[dict[str, Any]]] = {}
    for step in range(1, 11):
        target = step / 10.0
        selected_all: dict[str, dict[str, Any]] = {}
        for fold in fold_runtime:
            config = dict(fold["config"])
            config["selected_fraction"] = target
            selected_all.update(V23.select_actions(fold["test_rows"], fold["models"], config))
        summary, per = aggregate_selected(rows, selected_all)
        significance = V23.bootstrap(per, samples=samples, seed=seed + step)
        record = {
            "target_coverage": target,
            "realized_coverage": 1.0 - summary["fallback_rate"],
            "answer_f1_delta": summary["answer_f1_delta"],
            "title_support_recall_delta": summary["support_recall_delta"],
            "title_support_f1_delta": summary["sp_f1_delta"],
            "product_delta": summary["joint_f1_delta"],
            "selected_answer_drop_rate": summary["selected_answer_drop_rate"],
            "fallback_rate": summary["fallback_rate"],
        }
        for metric, prefix in [
            ("answer_f1", "answer_f1"),
            ("support_recall@5", "title_support_recall"),
            ("sp_f1", "title_support_f1"),
            ("joint_f1", "product"),
        ]:
            metric_result = significance["metrics"][metric]
            record[prefix + "_ci95_low"] = metric_result["ci95"][0]
            record[prefix + "_ci95_high"] = metric_result["ci95"][1]
            record[prefix + "_p_value"] = metric_result["p_value"]
        output_rows.append(record)
        per_by_coverage[f"{target:.1f}"] = per
    return output_rows, per_by_coverage


def write_risk_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_risk_coverage_figure(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"created": False, "error": f"{type(exc).__name__}: {exc}"}
    x = [row["realized_coverage"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(x, [r["answer_f1_delta"] for r in rows], marker="o", label="Answer F1")
    axes[0].plot(x, [r["product_delta"] for r in rows], marker="s", label="Answer-title product")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Realized action coverage")
    axes[0].set_ylabel("Held-out mean delta")
    axes[0].legend(frameon=False)
    axes[1].plot(x, [r["title_support_recall_delta"] for r in rows], marker="o", label="Title recall")
    axes[1].plot(x, [r["title_support_f1_delta"] for r in rows], marker="s", label="Title F1")
    axes[1].plot(x, [r["selected_answer_drop_rate"] for r in rows], marker="^", label="Selected answer-drop rate")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Realized action coverage")
    axes[1].set_ylabel("Delta / risk")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return {"created": True, "path": str(path)}


def action_scope_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    main_families = {"insert1", "bridge", "top4_bg1"}
    by_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_main: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family = defaultdict(lambda: {"actions": 0, "paper_positive": 0, "queries": set()})
    for row in rows:
        qid = row["query_id"]
        by_all[qid].append(row)
        fam = row.get("candidate_family", "unknown")
        family[fam]["actions"] += 1
        family[fam]["paper_positive"] += int(row.get("paper_positive", 0))
        family[fam]["queries"].add(qid)
        if fam in main_families:
            by_main[qid].append(row)
    all_positive = sum(int(row.get("paper_positive", 0)) for row in rows)
    main_rows = [row for row in rows if row.get("candidate_family") in main_families]
    main_positive = sum(int(row.get("paper_positive", 0)) for row in main_rows)
    return {
        "num_queries": len(by_all),
        "all_materialized": {
            "actions": len(rows),
            "paper_positive_actions": all_positive,
            "queries_with_no_paper_positive": sum(
                1 for values in by_all.values() if not any(v.get("paper_positive", 0) for v in values)
            ),
        },
        "main_eligible": {
            "families": sorted(main_families),
            "actions": len(main_rows),
            "paper_positive_actions": main_positive,
            "queries_with_no_paper_positive": sum(
                1 for qid in by_all if not any(v.get("paper_positive", 0) for v in by_main.get(qid, []))
            ),
        },
        "excluded_action_effect": {
            "positive_actions_removed": all_positive - main_positive,
            "additional_no_positive_queries": sum(
                1
                for qid, values in by_all.items()
                if any(v.get("paper_positive", 0) for v in values)
                and not any(v.get("paper_positive", 0) for v in by_main.get(qid, []))
            ),
        },
        "families": {
            name: {
                "actions": values["actions"],
                "queries": len(values["queries"]),
                "paper_positive_actions": values["paper_positive"],
            }
            for name, values in sorted(family.items())
        },
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.actions)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    primary = run_primary(rows)
    significance = V23.bootstrap(
        primary["per"], samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    write_json(output / "nested_final_1000_summary.json", primary["summary"])
    write_jsonl(output / "nested_per_example_delta.jsonl", primary["per"])
    write_json(output / "nested_significance_report.json", significance)
    write_json(output / "nested_fold_configs.json", primary["fold_records"])
    write_json(output / "nested_feature_audit.json", {"folds": primary["feature_audits"]})

    ablations = {
        "primary_nested_lexicographic": primary["summary"],
        "without_nested_safe_feature": run_nested_variant(
            rows,
            "without_nested_safe_feature",
            ["two_stage"],
            drop_nested_safety=True,
        ),
        "without_support_features": run_nested_variant(
            rows,
            "without_support_features",
            ["two_stage"],
            drop_support=True,
        ),
        "inherited_weighted_utility_diagnostic": run_nested_variant(
            rows,
            "inherited_weighted_utility_diagnostic",
            ["pairwise_ranker"],
            target_weights=(0.8, 0.3, 0.2),
        ),
    }
    write_json(output / "nested_ablation_summary.json", ablations)

    risk_rows, _ = risk_coverage(
        rows,
        primary["fold_runtime"],
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    write_risk_coverage_csv(output / "risk_coverage_curve.csv", risk_rows)
    figure_status = write_risk_coverage_figure(output / "risk_coverage_figure.pdf", risk_rows)
    write_json(output / "risk_coverage_figure_status.json", figure_status)

    if not args.skip_weight_sensitivity:
        weight_results = []
        for answer_w in (0.5, 0.8, 1.0):
            for support_w in (0.1, 0.3, 0.5):
                for title_w in (0.1, 0.2, 0.3):
                    weight_results.append(
                        run_nested_variant(
                            rows,
                            f"weights_a{answer_w}_s{support_w}_t{title_w}",
                            ["pairwise_ranker"],
                            target_weights=(answer_w, support_w, title_w),
                        )
                    )
        write_json(
            output / "utility_weight_sensitivity.json",
            {
                "selection_policy": "No held-out result is used to select weights; primary protocol is weight-free two-stage gating.",
                "variants": weight_results,
            },
        )

    write_json(output / "action_scope_statistics.json", action_scope_statistics(rows))
    print(json.dumps(primary["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
