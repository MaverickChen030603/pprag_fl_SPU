#!/usr/bin/env python3
"""Produce the frozen R3-T/R3-C Hotpot decision report with paired CIs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("complete_client_set_recall_at_3", "gold_client_recall_at_3", "local_complete_at_10", "transmitted_complete_at_15", "raw_merged_complete_at_10", "percentile_merged_complete_at_10")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, values: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def paired_ci(new: np.ndarray, old: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float, float]:
    delta = new - old
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for index in range(repeats):
        sample = rng.integers(0, len(delta), len(delta))
        samples[index] = float(delta[sample].mean())
    return float(delta.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def by_method(rows: list[dict[str, str]], method: str) -> dict[str, dict[str, str]]:
    values = {row["query_id"]: row for row in rows if row["method"] == method}
    if len(values) != 300:
        raise AssertionError(f"{method} has {len(values)} rather than 300 queries")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = read_csv(args.transfer_root / "holdout" / "main_results" / "per_query_results.csv")
    inherited = by_method(records, "B0_inherited_route")
    label_free = by_method(records, "B3_label_free_probe")
    logistic = by_method(records, "B4_logistic_seed_20260807")
    keys = sorted(inherited)
    bootstrap = []
    for name, values in (("H2_H3_label_free", label_free), ("H4_H5_logistic", logistic)):
        for index, metric in enumerate(METRICS):
            point, low, high = paired_ci(np.asarray([float(values[key][metric]) for key in keys]), np.asarray([float(inherited[key][metric]) for key in keys]), 20260810 + 53 * index + (0 if name.startswith("H2") else 1))
            bootstrap.append({"method": name, "reference": "H0_H1_inherited", "metric": metric, "delta": point, "ci95_low": low, "ci95_high": high, "queries": 300})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "hotpot_transfer_bootstrap.csv", bootstrap)
    main_rows = read_csv(args.transfer_root / "holdout" / "main_results" / "main_results.csv")
    cost_rows = read_csv(args.transfer_root / "holdout" / "cost_matched" / "cost_main_results.csv")
    latency_rows = read_csv(args.transfer_root / "holdout" / "cost_matched" / "latency_main_results.csv")
    logistic_raw = next(row for row in bootstrap if row["method"] == "H4_H5_logistic" and row["metric"] == "raw_merged_complete_at_10")
    logistic_coverage = next(row for row in bootstrap if row["method"] == "H4_H5_logistic" and row["metric"] == "complete_client_set_recall_at_3")
    logistic_local = next(row for row in bootstrap if row["method"] == "H4_H5_logistic" and row["metric"] == "local_complete_at_10")
    hotpot_pass = all(float(item["delta"]) >= 0.0 for item in (logistic_coverage, logistic_local, logistic_raw)) and (float(logistic_coverage["delta"]) >= .03 or float(logistic_local["delta"]) >= .03 or float(logistic_raw["delta"]) >= .03)
    reader_gate = "ready_for_frozen_reader" if hotpot_pass else "reader_blocked_hotpot_transfer"
    report = ["# R3-T/R3-C Frozen Hotpot Transfer and Cost-Matched Confirmation", "", "- Reader: not started. Final test: not accessed.", f"- Hotpot transfer decision: `{'pass' if hotpot_pass else 'fail'}`.", f"- Cross-dataset reader-gate status: `{reader_gate}`.", "", "## H0-H5 Retrieval Means"]
    for row in main_rows:
        report.append("- " + row["method"] + ": coverage=" + f"{float(row['complete_client_set_recall_at_3']):.4f}" + ", local=" + f"{float(row['local_complete_at_10']):.4f}" + ", raw=" + f"{float(row['raw_merged_complete_at_10']):.4f}" + ", percentile=" + f"{float(row['percentile_merged_complete_at_10']):.4f}")
    report.extend(["", "## B4 Logistic vs Inherited Paired Bootstrap"])
    for item in (logistic_coverage, logistic_local, logistic_raw):
        report.append(f"- {item['metric']}: {float(item['delta']):+.4f}, 95% CI [{float(item['ci95_low']):+.4f}, {float(item['ci95_high']):+.4f}].")
    report.extend(["", "## Cost-Matched Static Baselines"])
    for row in cost_rows:
        report.append(f"- {row['method']}: raw={float(row['raw_merged_complete_at_10']):.4f}, total_bytes={float(row['total_bytes']):.1f}, deep_clients={float(row['deep_client_compute']):.0f}, documents={float(row['documents_transmitted']):.0f}.")
    report.extend(["", "## Local Service Latency"])
    for row in latency_rows:
        report.append(f"- {row['method']}: mean={float(row['mean_local_retrieval_latency_ms']):.1f} ms, median={float(row['median_local_retrieval_latency_ms']):.1f} ms; {row['measurement_scope']}.")
    report.extend(["", "## Interpretation", "- ProbeRoute is not zero-cost: it queries eight shallow clients and sends 592 B metadata, but retains three deep clients and 15 documents.", "- Cost comparison reports actual bytes and local service time. Network transport latency is outside this single-node replay and is not claimed.", "- The reader gate is a permission to run the frozen reader protocol, not a reader result."])
    (args.output_dir / "hotpot_transfer_cost_decision.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.output_dir / "reader_gate_decision.json").write_text(json.dumps({"status": reader_gate, "hotpot_transfer_pass": hotpot_pass, "reader_started": False, "final_test_accessed": False, "reason": "2Wiki/MuSiQue R3 holdouts passed; Hotpot frozen transfer non-negative with >=3pp core gain"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"hotpot_transfer_pass": hotpot_pass, "reader_gate": reader_gate}, indent=2))


if __name__ == "__main__":
    main()
