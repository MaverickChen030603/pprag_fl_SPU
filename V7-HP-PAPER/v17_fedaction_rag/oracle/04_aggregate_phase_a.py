#!/usr/bin/env python3
"""Aggregate matched Phase-A cells and issue the preregistered machine decision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [array[rng.integers(0, len(array), len(array))].mean() for _ in range(samples)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_randomization_p(values: list[float], samples: int, seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or np.allclose(array, 0.0):
        return 1.0
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(samples):
        signs = rng.choice((-1.0, 1.0), size=len(array))
        exceed += abs(float((array * signs).mean())) >= observed
    return (exceed + 1.0) / (samples + 1.0)


def bh_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(values)
    adjusted = np.ones(len(values), dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, float(values[index]) * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--minimum-queries", type=int, default=100)
    args = parser.parse_args()

    rows = [row for path in sorted(args.input_dir.glob("*_per_query.csv")) for row in read_csv(path)]
    grouped: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["reader"], row["partition"], int(row["client_budget"]))].append(row)
    aggregate = []
    for key, values in sorted(grouped.items()):
        strict = [float(row["cross_client_strict_syn_joint_f1"]) for row in values]
        only = [int(row["cross_client_composition_only_joint_f1"]) for row in values]
        low, high = ci(strict, args.bootstrap, args.seed)
        dispersed = [int(row["cross_client_composition_only_joint_f1"]) for row in values if row["cross_client_evidence"] == "1"]
        local = [int(row["cross_client_composition_only_joint_f1"]) for row in values if row["cross_client_evidence"] == "0"]
        aggregate.append({
            "dataset": key[0], "reader": key[1], "partition": key[2], "client_budget": key[3], "queries": len(values),
            "mean_cross_client_strict_syn_joint": float(np.mean(strict)), "strict_syn_ci_low": low, "strict_syn_ci_high": high,
            "strict_syn_two_sided_p": paired_randomization_p(strict, args.bootstrap, args.seed + 17),
            "cross_client_composition_only_rate": float(np.mean(only)),
            "composition_only_rate_dispersed": float(np.mean(dispersed)) if dispersed else "",
            "composition_only_rate_not_dispersed": float(np.mean(local)) if local else "",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "federated_oracle_results.csv", aggregate)
    write_csv(args.output_dir / "cross_client_synergy.csv", aggregate)
    write_csv(args.output_dir / "cross_client_only_rates.csv", aggregate)

    by_condition = {(row["dataset"], row["reader"], row["partition"], int(row["client_budget"]), row["query_id"]): row for row in rows}
    controls = []
    for natural in ("topic_silo", "entity_community"):
        for dataset in sorted({row["dataset"] for row in rows}):
            for reader in sorted({row["reader"] for row in rows}):
                natural_rows = [row for row in rows if row["dataset"] == dataset and row["reader"] == reader and row["partition"] == natural and int(row["client_budget"]) == 3]
                for control in ("random_control", "centralized"):
                    diffs_only, diffs_syn = [], []
                    for row in natural_rows:
                        match = by_condition.get((dataset, reader, control, 3 if control != "centralized" else 1, row["query_id"]))
                        if match is None:
                            continue
                        diffs_only.append(int(row["cross_client_composition_only_joint_f1"]) - int(match["cross_client_composition_only_joint_f1"]))
                        diffs_syn.append(float(row["cross_client_strict_syn_joint_f1"]) - float(match["cross_client_strict_syn_joint_f1"]))
                    if diffs_only:
                        only_low, only_high = ci(diffs_only, args.bootstrap, args.seed)
                        syn_low, syn_high = ci(diffs_syn, args.bootstrap, args.seed + 1)
                        controls.append({
                            "dataset": dataset, "reader": reader, "natural_partition": natural, "control": control, "matched_queries": len(diffs_only),
                            "mean_composition_only_rate_difference": float(np.mean(diffs_only)), "composition_only_diff_ci_low": only_low, "composition_only_diff_ci_high": only_high,
                            "composition_only_two_sided_p": paired_randomization_p(diffs_only, args.bootstrap, args.seed + 23),
                            "mean_strict_syn_difference": float(np.mean(diffs_syn)), "strict_syn_diff_ci_low": syn_low, "strict_syn_diff_ci_high": syn_high,
                            "strict_syn_two_sided_p": paired_randomization_p(diffs_syn, args.bootstrap, args.seed + 29),
                        })
    for field in ("composition_only_two_sided_p", "strict_syn_two_sided_p"):
        adjusted = bh_adjust([float(row[field]) for row in controls])
        for row, value in zip(controls, adjusted):
            row[field.replace("_p", "_bh_p")] = value
    write_csv(args.output_dir / "partition_control_results.csv", controls)

    gap_fields = ("routing_absence", "local_retrieval_absence", "pool_absence", "single_action_absence", "cross_client_composition_absence", "composition_search_miss")
    gap_rows = []
    for key, values in sorted(grouped.items()):
        if key[2] == "centralized":
            continue
        gap_rows.append({"dataset": key[0], "reader": key[1], "partition": key[2], "client_budget": key[3], "queries": len(values), **{
            field + "_rate": float(np.mean([int(row[field]) for row in values if row[field] != ""])) for field in gap_fields
        }})
    write_csv(args.output_dir / "federated_opportunity_gap.csv", gap_rows)

    datasets = sorted({row["dataset"] for row in aggregate})
    readers = sorted({row["reader"] for row in aggregate})
    required_partition_budgets = (
        ("centralized", 1),
        ("topic_silo", 2),
        ("topic_silo", 3),
        ("entity_community", 3),
        ("random_control", 3),
    )
    required_cells = {
        (dataset, reader, partition, budget)
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
        for reader in ("flan", "unifiedqa")
        for partition, budget in required_partition_budgets
    }
    observed_counts = {
        (row["dataset"], row["reader"], row["partition"], int(row["client_budget"])): int(row["queries"])
        for row in aggregate
    }
    missing_cells = sorted(cell for cell in required_cells if cell not in observed_counts)
    undersized_cells = sorted(
        cell for cell in required_cells
        if cell in observed_counts and observed_counts[cell] < args.minimum_queries
    )
    enough = not missing_cells and not undersized_cells
    dataset_pass: dict[str, bool] = {}
    details = {}
    for dataset in datasets:
        cell_passes = []
        for reader in readers:
            cell = next((row for row in aggregate if row["dataset"] == dataset and row["reader"] == reader and row["partition"] == "topic_silo" and row["client_budget"] == 3), None)
            budget_two = next((row for row in aggregate if row["dataset"] == dataset and row["reader"] == reader and row["partition"] == "topic_silo" and row["client_budget"] == 2), None)
            random_cmp = next((row for row in controls if row["dataset"] == dataset and row["reader"] == reader and row["natural_partition"] == "topic_silo" and row["control"] == "random_control"), None)
            central_cmp = next((row for row in controls if row["dataset"] == dataset and row["reader"] == reader and row["natural_partition"] == "topic_silo" and row["control"] == "centralized"), None)
            dispersed = cell and cell["composition_only_rate_dispersed"] != "" and cell["composition_only_rate_not_dispersed"] != "" and float(cell["composition_only_rate_dispersed"]) > float(cell["composition_only_rate_not_dispersed"])
            budget_two_positive = bool(
                budget_two
                and budget_two["queries"] >= args.minimum_queries
                and budget_two["mean_cross_client_strict_syn_joint"] > 0
                and budget_two["cross_client_composition_only_rate"] > 0
            )
            passed = bool(cell and cell["queries"] >= args.minimum_queries and cell["strict_syn_ci_low"] > 0 and cell["cross_client_composition_only_rate"] >= 0.10 and random_cmp and random_cmp["composition_only_diff_ci_low"] > 0 and central_cmp and central_cmp["composition_only_diff_ci_low"] > 0 and dispersed and budget_two_positive)
            cell_passes.append(passed)
        dataset_pass[dataset] = len(cell_passes) == 2 and all(cell_passes)
        details[dataset] = cell_passes
    passed_datasets = sum(dataset_pass.values())
    status = "insufficient_sample" if not enough else ("continue_phase_b" if passed_datasets >= 2 else "hold_or_redirect")
    decision = {
        "status": status,
        "minimum_queries": args.minimum_queries,
        "dataset_pass": dataset_pass,
        "reader_cell_passes": details,
        "passed_datasets": passed_datasets,
        "required_cells": len(required_cells),
        "missing_cells": missing_cells,
        "undersized_cells": undersized_cells,
        "primary_partition": "topic_silo",
        "rule": "At least two datasets must pass on both readers with Bc=3 topic-silo StrictSyn CI>0, composition-only>=10%, paired rate differences over random and centralized controls CI>0, higher opportunity on dispersed-evidence queries, and positive Bc=2 secondary evidence.",
    }
    (args.output_dir / "federated_go_no_go_phase_a.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "federated_go_no_go_phase_a.md").write_text(
        "# V17 Federated Oracle Go/No-Go\n\n" + f"Status: **{status}**\n\n```json\n" + json.dumps(decision, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
