#!/usr/bin/env python3
"""Aggregate frozen R4 reader outputs with pre-registered paired analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METHODS = ("federated_baseline", "label_free_proberoute", "logistic_proberoute", "centralized_retrieval_reference")
COMPARISONS = (("label_free_proberoute", "federated_baseline"), ("logistic_proberoute", "federated_baseline"), ("logistic_proberoute", "label_free_proberoute"))
METRICS = ("answer_f1", "sp_f1", "joint_f1", "answer_em", "sp_em", "joint_em")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in data for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(data)


def bootstrap(delta: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    values = np.asarray([delta[rng.integers(0, n, n)].mean() for _ in range(5000)])
    return float(delta.mean()), float(np.quantile(values, .025)), float(np.quantile(values, .975))


def bh(rows_: list[dict[str, Any]]) -> None:
    candidates = [r for r in rows_ if r["metric"] != "joint_f1"]
    ordered = sorted(enumerate(candidates), key=lambda x: x[1]["two_sided_p"])
    m = len(ordered); running = 1.0
    for rank, (_, row) in reversed(list(enumerate([x[1] for x in ordered], start=1))):
        running = min(running, row["two_sided_p"] * m / rank)
        row["bh_fdr_q"] = running
    for row in rows_:
        row.setdefault("bh_fdr_q", "not_applicable_primary" if row["metric"] == "joint_f1" else "")


def support_transition(base: int, probe: int) -> str:
    return {(0, 1): "T1_rescue", (1, 1): "T2_preserved_complete", (1, 0): "T3_harm", (0, 0): "T4_persistent_incomplete"}[base, probe]


def decision(results: list[dict[str, Any]], comparisons: list[dict[str, Any]], mechanism: list[dict[str, Any]]) -> str:
    # Pre-registered primary assessment: R2 Joint F1 versus R0, averaging
    # readers within each dataset.  Reader-cell CIs remain the strong-pass test.
    primary = [r for r in comparisons if r["method_a"] == "logistic_proberoute" and r["method_b"] == "federated_baseline" and r["metric"] == "joint_f1"]
    average_by_dataset = defaultdict(list)
    for row in primary: average_by_dataset[row["dataset"]].append(float(row["mean_delta"]))
    positive_datasets = sum(np.mean(values) > 0 for values in average_by_dataset.values())
    cells_ci_positive = sum(float(row["ci_low"]) > 0 for row in primary)
    negative_both = any(all(value < 0 for value in values) for values in average_by_dataset.values())
    rescue = [r for r in mechanism if r["comparison"] == "logistic_proberoute_vs_federated_baseline" and r["support_transition"] == "T1_rescue"]
    rescue_positive = bool(rescue) and np.mean([float(r["delta_joint_f1"]) for r in rescue]) > 0
    answer_harm = any(r["metric"] == "answer_f1" and float(r["mean_delta"]) < 0 for r in comparisons
                      if r["method_a"] == "logistic_proberoute" and r["method_b"] == "federated_baseline")
    if positive_datasets >= 2 and cells_ci_positive >= 2 and not negative_both and rescue_positive and not answer_harm:
        return "probe_route_end_to_end_confirmed"
    if positive_datasets >= 2 and not negative_both:
        return "retrieval_confirmed_reader_gain_weak"
    if negative_both or answer_harm:
        return "reader_context_misalignment_detected"
    return "retrieval_method_confirmed_reader_bottleneck"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flan", type=Path, required=True)
    parser.add_argument("--unifiedqa", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    flan_rows, unified_rows = list(rows(args.flan)), list(rows(args.unifiedqa))
    data = flan_rows + unified_rows
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in data:
        grouped[(row["dataset"], row["reader"], row["query_id"])][row["method"]] = row
    invalid = [key for key, values in grouped.items() if set(values) != set(METHODS)]
    if invalid:
        raise ValueError(f"incomplete method set for {len(invalid)} queries; first={invalid[:2]}")
    main_rows = []
    for dataset, reader, _ in sorted(grouped):
        cell = [values for (ds, rd, _), values in grouped.items() if ds == dataset and rd == reader]
        for method in METHODS:
            main_rows.append({"dataset": dataset, "reader": reader, "method": method, "queries": len(cell), **{metric: float(np.mean([float(row[method][metric]) for row in cell])) for metric in METRICS}, "reader_context_complete_support": float(np.mean([float(row[method]["retrieval_complete_support"]) for row in cell]))})
    comparisons = []
    for (dataset, reader, _), values in sorted(grouped.items()):
        pass
    cells = sorted({(key[0], key[1]) for key in grouped})
    for dataset, reader in cells:
        entries = [values for (ds, rd, _), values in grouped.items() if ds == dataset and rd == reader]
        for method_a, method_b in COMPARISONS:
            for metric in ("answer_f1", "sp_f1", "joint_f1"):
                delta = np.asarray([float(row[method_a][metric]) - float(row[method_b][metric]) for row in entries])
                mean, low, high = bootstrap(delta, int(hashlib.sha256(f"{dataset}|{reader}|{method_a}|{method_b}|{metric}".encode()).hexdigest()[:8], 16))
                comparisons.append({"dataset": dataset, "reader": reader, "comparison": f"{method_a}_vs_{method_b}", "method_a": method_a, "method_b": method_b, "metric": metric, "queries": len(delta), "mean_delta": mean, "ci_low": low, "ci_high": high, "paired_win": int((delta > 0).sum()), "paired_tie": int((delta == 0).sum()), "paired_loss": int((delta < 0).sum()), "two_sided_p": float(2 * min((delta <= 0).mean(), (delta >= 0).mean()))})
    bh(comparisons)
    transitions, context_change = [], []
    for dataset, reader in cells:
        entries = [(qid, values) for (ds, rd, qid), values in grouped.items() if ds == dataset and rd == reader]
        for probe in ("label_free_proberoute", "logistic_proberoute"):
            name = f"{probe}_vs_federated_baseline"
            buckets = defaultdict(list)
            changes = defaultdict(list)
            for _, values in entries:
                base, candidate = values["federated_baseline"], values[probe]
                state = support_transition(int(base["retrieval_complete_support"]), int(candidate["retrieval_complete_support"]))
                buckets[state].append((base, candidate))
                identical = base["context_hash"] == candidate["context_hash"]
                if identical: category = "context_byte_identical"
                elif state == "T1_rescue": category = "support_rescue_context_changed"
                elif state == "T3_harm": category = "support_harm"
                elif state == "T2_preserved_complete": category = "support_preserved_non_support_swap"
                else: category = "context_changed_support_unchanged"
                changes[category].append((base, candidate))
            for state, pairs in buckets.items():
                transitions.append({"dataset": dataset, "reader": reader, "comparison": name, "support_transition": state, "n": len(pairs), "delta_answer_f1": float(np.mean([float(b["answer_f1"]) - float(a["answer_f1"]) for a, b in pairs])), "delta_sp_f1": float(np.mean([float(b["sp_f1"]) - float(a["sp_f1"]) for a, b in pairs])), "delta_joint_f1": float(np.mean([float(b["joint_f1"]) - float(a["joint_f1"]) for a, b in pairs]))})
            for category, pairs in changes.items():
                context_change.append({"dataset": dataset, "reader": reader, "comparison": name, "context_category": category, "n": len(pairs), "rate": len(pairs) / len(entries), "answer_change_rate": float(np.mean([float(a["answer_f1"]) != float(b["answer_f1"]) for a, b in pairs])), "reader_positive_change_rate": float(np.mean([float(b["joint_f1"]) > float(a["joint_f1"]) for a, b in pairs])), "reader_negative_change_rate": float(np.mean([float(b["joint_f1"]) < float(a["joint_f1"]) for a, b in pairs]))})
    gap = []
    for dataset, reader in cells:
        lookup = {row["method"]: row for row in main_rows if row["dataset"] == dataset and row["reader"] == reader}
        for method in ("label_free_proberoute", "logistic_proberoute"):
            for metric in ("reader_context_complete_support", "sp_f1", "joint_f1"):
                denominator = float(lookup["centralized_retrieval_reference"][metric]) - float(lookup["federated_baseline"][metric])
                gap.append({"dataset": dataset, "reader": reader, "method": method, "metric": metric, "gap_recovery": "N/A" if denominator <= 0 else (float(lookup[method][metric]) - float(lookup["federated_baseline"][metric])) / denominator})
    out = args.output_root
    write_csv(out / "statistics/main_reader_results.csv", main_rows)
    write_csv(out / "flan/per_query_results.csv", flan_rows)
    write_csv(out / "unifiedqa/per_query_results.csv", unified_rows)
    write_csv(out / "statistics/paired_bootstrap.csv", comparisons)
    write_csv(out / "statistics/bh_secondary_tests.csv", comparisons)
    write_csv(out / "mechanism/support_transition_analysis.csv", transitions)
    write_csv(out / "mechanism/reader_gain_given_support_rescue.csv", [row for row in transitions if row["support_transition"] == "T1_rescue"])
    write_csv(out / "mechanism/context_change_analysis.csv", context_change)
    write_csv(out / "gap_recovery/gap_recovery.csv", gap)
    enriched = []
    for (dataset, reader, query_id), values in sorted(grouped.items()):
        base = values["federated_baseline"]
        for method, row in values.items():
            copy = dict(row)
            copy["baseline_retrieval_complete_support"] = base["retrieval_complete_support"]
            copy["support_state_transition"] = support_transition(int(base["retrieval_complete_support"]), int(row["retrieval_complete_support"]))
            enriched.append(copy)
    write_csv(out / "statistics/per_query_results.csv", enriched)
    final = decision(main_rows, comparisons, transitions)
    report = ["# V20 R4 Frozen Dual-Reader Go/No-Go", "", f"**Final status:** `{final}`", "", "R4 evaluated frozen R3 contexts only. The centralized retrieval reference is a same-contract reference, not an upper bound.", "", "## Primary Joint F1 comparisons"]
    for row in comparisons:
        if row["metric"] == "joint_f1" and row["method_a"] == "logistic_proberoute" and row["method_b"] == "federated_baseline":
            report.append(f"- {row['dataset']} / {row['reader']}: {row['mean_delta']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}], W/T/L={row['paired_win']}/{row['paired_tie']}/{row['paired_loss']}")
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports/r4_reader_go_no_go.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "reports/r5_final_test_recommendation.md").write_text(f"# R5 recommendation\n\nR4 status: `{final}`. Final test remains sealed; R4 does not unseal it.\n", encoding="utf-8")
    (out / "reports/r4_full_experimental_report.md").write_text(
        "# V20 Stage R4: Frozen Dual-Reader End-to-End Evaluation\n\n"
        f"**Status:** `{final}`\n\n"
        "This report evaluates only pre-materialized R3 contexts. It uses the legacy frozen Top-10/Top-5 reader contract, deterministic decoding, and query-level paired bootstrap (5,000 resamples). The centralized retrieval reference is a reference comparator, not a mathematical upper bound.\n\n"
        "## Artifacts\n\n"
        "- `statistics/main_reader_results.csv`: all formal dataset-reader-method means.\n"
        "- `statistics/paired_bootstrap.csv`: R1-R0, R2-R0, R2-R1 paired effects.\n"
        "- `mechanism/support_transition_analysis.csv`: evidence rescue/preservation/harm transmission.\n"
        "- `gap_recovery/gap_recovery.csv`: centralized-reference gap recovery where the denominator is positive.\n",
        encoding="utf-8")
    print(json.dumps({"status": final, "cells": len(cells), "main": str(out / "statistics/main_reader_results.csv")}, indent=2))


if __name__ == "__main__":
    main()
