#!/usr/bin/env python3
"""Materialize V18's unified report from a completed, audited V17 Phase-A."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
READERS = ("flan", "unifiedqa")
CONDITIONS = {
    "centralized": ("centralized", 1),
    "topic_silo_bc2": ("topic_silo", 2),
    "topic_silo": ("topic_silo", 3),
    "entity_community": ("entity_community", 3),
    "random_control": ("random_control", 3),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, str]], field: str, predicate=lambda _: True) -> float | str:
    values = [float(row[field]) for row in rows if row.get(field, "") != "" and predicate(row)]
    return float(np.mean(values)) if values else ""


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = [array[rng.integers(0, len(array), len(array))].mean() for _ in range(samples)]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_p(values: list[float], samples: int, seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or np.allclose(array, 0.0):
        return 1.0
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    exceeded = sum(abs(float((array * rng.choice((-1.0, 1.0), len(array))).mean())) >= observed for _ in range(samples))
    return (exceeded + 1.0) / (samples + 1.0)


def control_map(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {(row["dataset"], row["reader"], row["natural_partition"], row["control"]): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    integrity = json.loads(args.integrity.read_text(encoding="utf-8"))
    if integrity.get("status") != "pass":
        raise SystemExit("Refusing report generation: Checkpoint-A integrity did not pass.")
    results = args.phase_dir / "results"
    routing_rows = read_csv(results / "routing_metrics_summary.csv")
    routing = {(r["dataset"], r["partition"], int(r["client_budget"])): r for r in routing_rows}
    controls = control_map(read_csv(results / "partition_control_results.csv"))
    cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for condition, (partition, budget) in CONDITIONS.items():
            route = routing.get((dataset, partition, budget), {})
            for reader in READERS:
                path = results / f"{dataset}_{condition}_{reader}_per_query.csv"
                rows = read_csv(path)
                strict = [float(row["cross_client_strict_syn_joint_f1"]) for row in rows]
                low, high = bootstrap_ci(strict, args.bootstrap, args.seed)
                cross = [row for row in rows if row.get("cross_client_evidence") == "1"]
                feasible = [row for row in rows if row.get("pool_absence") == "0"]
                random = controls.get((dataset, reader, partition, "random_control"), {})
                central = controls.get((dataset, reader, partition, "centralized"), {})
                only = [int(row["cross_client_composition_only_joint_f1"]) for row in rows]
                unit_positive = low > 0 and float(np.mean(only)) >= 0.10
                cells.append({
                    "dataset": dataset, "reader": reader, "partition": partition,
                    "N": len(rows), "Bc": budget, "local_k": 5, "global_pool": 10,
                    "cross_client_evidence_rate": len(cross) / len(rows) if partition != "centralized" else "",
                    "pool_complete_support_rate": route.get("mean_complete_support_in_action_pool", ""),
                    "best_single_client_delta": mean(rows, "best_single_client_delta_joint_f1"),
                    "best_flat_union_delta": "not_available_in_frozen_v17_contract",
                    "best_single_cross_edit_delta": mean(rows, "best_single_cross_delta_joint_f1"),
                    "best_cross_composition_delta": mean(rows, "best_cross_composition_delta_joint_f1"),
                    "CrossClientStrictSyn": float(np.mean(strict)), "strict_syn_ci_low": low,
                    "strict_syn_ci_high": high, "paired_p": paired_p(strict, args.bootstrap, args.seed + 19),
                    "composition_only_count": int(sum(only)), "composition_only_rate_all": float(np.mean(only)),
                    "composition_only_rate_cross_client_evidence": mean(cross, "cross_client_composition_only_joint_f1"),
                    "composition_only_rate_pool_feasible": mean(feasible, "cross_client_composition_only_joint_f1"),
                    "centralized_difference": central.get("mean_composition_only_rate_difference", ""),
                    "random_control_difference": random.get("mean_composition_only_rate_difference", ""),
                    "mean_clients_used": route.get("mean_clients_contacted", ""),
                    "documents_transferred": route.get("mean_documents_transmitted", ""),
                    "go_no_go_unit": "positive_candidate" if unit_positive else "not_positive",
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "checkpoint_a_all_units.csv", cells)

    reader_consistency: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        grouped[(row["dataset"], row["partition"], int(row["Bc"]))].append(row)
    for (dataset, partition, budget), values in sorted(grouped.items()):
        signs = [float(row["CrossClientStrictSyn"]) > 0 for row in values]
        reader_consistency.append({
            "dataset": dataset, "partition": partition, "Bc": budget,
            "readers": "|".join(sorted(row["reader"] for row in values)),
            "same_positive_direction": len(signs) == 2 and all(signs),
            "strict_syn_range": max(float(row["CrossClientStrictSyn"]) for row in values) - min(float(row["CrossClientStrictSyn"]) for row in values),
        })
    write_csv(args.output_dir / "checkpoint_a_reader_consistency.csv", reader_consistency)
    partition_comparison = read_csv(results / "partition_control_results.csv")
    write_csv(args.output_dir / "checkpoint_a_partition_comparison.csv", partition_comparison)

    v17_decision = json.loads((results / "federated_go_no_go_phase_a.json").read_text(encoding="utf-8"))
    passes = int(v17_decision.get("passed_datasets", 0))
    branch = "checkpoint_a_strong_pass" if passes >= 2 else ("checkpoint_a_conditional_pass" if passes == 1 else "checkpoint_a_fail")
    go = {"v17_machine_decision": v17_decision.get("status"), "v18_branch": branch, "passed_datasets": passes}
    (args.output_dir / "checkpoint_a_go_no_go.json").write_text(json.dumps(go, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "checkpoint_a_go_no_go.md").write_text("# Checkpoint-A Go/No-Go\n\n```json\n" + json.dumps(go, indent=2) + "\n```\n", encoding="utf-8")
    summary = ["# Checkpoint-A Summary", "", f"Machine branch: **{branch}**.", "", "## Required artifacts", "", f"- all units: `{args.output_dir / 'checkpoint_a_all_units.csv'}`", f"- reader consistency: `{args.output_dir / 'checkpoint_a_reader_consistency.csv'}`", f"- controls: `{args.output_dir / 'checkpoint_a_partition_comparison.csv'}`"]
    (args.output_dir / "checkpoint_a_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
