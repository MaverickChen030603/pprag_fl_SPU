#!/usr/bin/env python3
"""Summarize pre-registered R3 logistic-ranker holdout results and paired CIs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEEDS = (20260807, 20260808, 20260809)
METRICS = (
    "complete_client_set_recall_at_3",
    "gold_client_recall_at_3",
    "local_complete_at_10",
    "transmitted_complete_at_15",
    "raw_merged_complete_at_10",
    "percentile_merged_complete_at_10",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def bootstrap_delta(new: np.ndarray, old: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float, float]:
    if new.shape != old.shape:
        raise ValueError("paired vectors differ in length")
    point = float(np.mean(new - old))
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.integers(0, len(new), len(new))
        samples[index] = np.mean(new[sampled] - old[sampled])
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def method_rows(records: list[dict[str, str]], method: str) -> dict[str, dict[str, str]]:
    values = {row["query_id"]: row for row in records if row["method"] == method}
    if len(values) != 300:
        raise AssertionError(f"{method} has {len(values)} rather than 300 holdout rows")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_bootstrap, summary, decisions = [], [], []
    for dataset in ("2wikimultihopqa", "musique"):
        records = read_csv(args.stage_root / "holdout" / dataset / "main_results" / "per_query_results.csv")
        baseline = method_rows(records, "B1_static_p0")
        label_free = method_rows(records, "B3_label_free_probe")
        for method, comparator in (("B3_label_free_probe", baseline), *[(f"B4_logistic_seed_{seed}", baseline) for seed in SEEDS]):
            candidate = method_rows(records, method)
            if set(candidate) != set(comparator):
                raise AssertionError("paired query IDs differ")
            for metric_index, metric in enumerate(METRICS):
                keys = sorted(candidate)
                point, low, high = bootstrap_delta(
                    np.asarray([float(candidate[key][metric]) for key in keys]),
                    np.asarray([float(comparator[key][metric]) for key in keys]),
                    20260807 + metric_index + 101 * len(all_bootstrap),
                )
                all_bootstrap.append({
                    "dataset": dataset, "method": method, "reference": "B1_static_p0", "metric": metric,
                    "delta": point, "ci95_low": low, "ci95_high": high, "queries": len(keys),
                })
        for method in ("B1_static_p0", "B3_label_free_probe", *[f"B4_logistic_seed_{seed}" for seed in SEEDS]):
            values = method_rows(records, method)
            summary.append({"dataset": dataset, "method": method, "queries": len(values), **{
                metric: float(np.mean([float(row[metric]) for row in values.values()])) for metric in METRICS
            }})
        seed_rows = [row for row in all_bootstrap if row["dataset"] == dataset and row["method"].startswith("B4_")]
        by_seed = {seed: {row["metric"]: row for row in seed_rows if row["method"] == f"B4_logistic_seed_{seed}"} for seed in SEEDS}
        seed_passes = []
        for seed, result in by_seed.items():
            coverage = result["complete_client_set_recall_at_3"]
            evidence = result["local_complete_at_10"]
            merged = result["percentile_merged_complete_at_10"]
            all_non_degrade = all(row["delta"] >= -0.02 for row in result.values())
            seed_passes.append(
                coverage["delta"] >= 0.08 and evidence["delta"] >= 0.05 and merged["delta"] >= 0.05
                and coverage["ci95_low"] > 0.0 and (evidence["ci95_low"] > 0.0 or merged["ci95_low"] > 0.0)
                and all_non_degrade
            )
        decisions.append({"dataset": dataset, "all_three_seed_success": all(seed_passes), "seed_successes": int(sum(seed_passes)),
                          "reference": "B1_static_p0", "reader_started": False, "final_test_accessed": False})
    all_datasets_pass = all(item["all_three_seed_success"] for item in decisions)
    status = "probe_routing_method_confirmed" if all_datasets_pass else "probe_ranker_holdout_not_confirmed"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "main_results.csv", summary)
    write_csv(args.output_dir / "bootstrap_results.csv", all_bootstrap)
    report = ["# R3 Lightweight Probe Ranker Fresh-Holdout Report", "", f"- Status: `{status}`", "- Holdout: sealed R2-A.6 Recovery-Holdout, N=300 per dataset.", "- Reader: not started. Final test: not accessed.", "- Comparator: B1 static P0 Top-3. B3 is the frozen dataset-specific label-free probe comparator.", "", "## Pre-registered Decision"]
    for item in decisions:
        report.append(f"- {item['dataset']}: three-seed success {item['seed_successes']}/3; all-three={item['all_three_seed_success']}.")
    report.extend(["", "## Main Means"])
    for row in summary:
        report.append("- " + row["dataset"] + " / " + row["method"] + ": " + ", ".join(f"{metric}={row[metric]:.4f}" for metric in METRICS))
    (args.output_dir / "ranker_holdout_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.output_dir / "decision.json").write_text(json.dumps({"status": status, "per_dataset": decisions, "reader_started": False, "final_test_accessed": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "per_dataset": decisions}, indent=2))


if __name__ == "__main__":
    main()
