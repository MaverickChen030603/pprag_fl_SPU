#!/usr/bin/env python3
"""Analyze one V17 dataset-reader-partition Oracle cell."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRICS = ("answer_f1", "sp_f1", "joint_f1")


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def best_delta(rows: list[dict[str, Any]], baseline: float, predicate, metric: str) -> tuple[float, str]:
    eligible = [row for row in rows if predicate(row)]
    if not eligible:
        return 0.0, ""
    best = max(eligible, key=lambda row: (float(row[metric]), str(row["trajectory_id"])))
    return float(best[metric]) - baseline, str(best["trajectory_id"])


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = array[rng.integers(0, len(array), len(array))].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--dispersion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    metadata = {str(row["trajectory_id"]): row for row in jsonl(args.contexts)}
    pools = {str(row["query_id"]): row for row in jsonl(args.pool)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in jsonl(args.outcomes):
        spec = metadata[str(outcome["trajectory_id"])]
        grouped[str(outcome["query_id"])].append({**outcome, **{
            "partition": spec["partition"],
            "origin_client": spec["origin_client"],
            "selected_clients": spec["selected_clients"],
            "context_client_ids": spec["context_client_ids"],
            "distinct_clients": spec["distinct_clients"],
            "cross_client_docs": spec["cross_client_docs"],
            "is_single_cross_action": spec["is_single_cross_action"],
            "is_cross_composition": spec["is_cross_composition"],
            "is_within_client_composition": spec["is_within_client_composition"],
            "single_client_id": spec.get("single_client_id", ""),
            "client_budget": spec["client_budget"],
            "local_k": spec["local_k"],
        }})
    first = next(iter(grouped.values()))[0]
    dataset, reader, partition = first["dataset"], first["reader"], first["partition"]
    dispersion = {
        row["query_id"]: row
        for row in csv_rows(args.dispersion)
        if row["dataset"] == dataset and row["partition"] == partition
    }
    if partition == "centralized":
        dispersion = {}

    output_rows = []
    for query_id, outcomes in sorted(grouped.items()):
        baseline_row = next((row for row in outcomes if row.get("is_baseline")), None)
        if baseline_row is None:
            raise ValueError(f"{query_id}: no baseline outcome")
        row_out: dict[str, Any] = {
            "dataset": dataset,
            "reader": reader,
            "partition": partition,
            "query_id": query_id,
            "client_budget": int(baseline_row["client_budget"]),
            "local_k": int(baseline_row["local_k"]),
            "origin_client": int(baseline_row["origin_client"]),
        }
        for metric in METRICS:
            baseline = float(baseline_row[metric])
            if partition == "centralized":
                single_delta, single_id = best_delta(outcomes, baseline, lambda item: int(item["depth"]) == 1, metric)
                composed_delta, composed_id = best_delta(outcomes, baseline, lambda item: 2 <= int(item["depth"]) <= 3, metric)
                single_client_delta = 0.0
                single_cross_delta = single_delta
                fed_gain = composed_delta - single_delta
                composition_only = int(composed_delta > 0 and single_delta <= 0)
            else:
                single_client_delta, _ = best_delta(outcomes, baseline, lambda item: item.get("candidate_type") == "single_client_context" or item.get("is_baseline"), metric)
                single_cross_delta, single_id = best_delta(outcomes, baseline, lambda item: bool(item["is_single_cross_action"]), metric)
                composed_delta, composed_id = best_delta(outcomes, baseline, lambda item: bool(item["is_cross_composition"]), metric)
                comparator = max(single_client_delta, single_cross_delta)
                fed_gain = composed_delta - comparator
                composition_only = int(composed_delta > 0 and single_client_delta <= 0 and single_cross_delta <= 0)
            within_delta, _ = best_delta(outcomes, baseline, lambda item: bool(item["is_within_client_composition"]), metric)
            row_out.update({
                f"baseline_{metric}": baseline,
                f"best_single_client_delta_{metric}": single_client_delta,
                f"best_single_cross_delta_{metric}": single_cross_delta,
                f"best_cross_composition_delta_{metric}": composed_delta,
                f"cross_client_strict_syn_{metric}": fed_gain,
                f"within_client_best_composed_delta_{metric}": within_delta,
                f"cross_client_composition_only_{metric}": composition_only,
                f"best_single_cross_id_{metric}": single_id,
                f"best_cross_composition_id_{metric}": composed_id,
            })
        pool = pools[query_id]
        if partition != "centralized" and query_id in dispersion:
            audit = dispersion[query_id]
            support_clients = {int(value) for value in audit["support_clients"].split("|") if value}
            support_docs = {value for value in audit["support_doc_ids"].split("|") if value}
            selected_clients = set(map(int, pool["selected_clients"]))
            returned_docs = {doc["doc_id"] for doc in pool["pool"] if int(doc["client_id"]) in selected_clients}
            action_docs = {doc["doc_id"] for doc in pool["pool"][: int(pool["pool_size"])]}
            row_out.update({
                "cross_client_evidence": int(audit["cross_client_evidence"]),
                "support_client_count": int(audit["support_client_count"]),
                "routing_absence": int(not support_clients.issubset(selected_clients)),
                "local_retrieval_absence": int(bool(support_clients & selected_clients) and not (support_docs & returned_docs)),
                "pool_absence": int(not support_docs.issubset(action_docs)),
                "single_action_absence": int(max(row_out["best_single_client_delta_joint_f1"], row_out["best_single_cross_delta_joint_f1"]) <= 0),
                "cross_client_composition_absence": int(row_out["best_cross_composition_delta_joint_f1"] <= 0),
                "composition_search_miss": 0,
            })
        else:
            row_out.update({key: "" for key in ("cross_client_evidence", "support_client_count", "routing_absence", "local_retrieval_absence", "pool_absence", "single_action_absence", "cross_client_composition_absence", "composition_search_miss")})
        output_rows.append(row_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {"dataset": dataset, "reader": reader, "partition": partition, "queries": len(output_rows), "metrics": {}}
    for metric in METRICS:
        strict = [float(row[f"cross_client_strict_syn_{metric}"]) for row in output_rows]
        only = [int(row[f"cross_client_composition_only_{metric}"]) for row in output_rows]
        low, high = bootstrap_ci(strict, args.bootstrap, args.seed)
        summary["metrics"][metric] = {
            "mean_cross_client_strict_syn": float(np.mean(strict)),
            "strict_syn_ci_low": low,
            "strict_syn_ci_high": high,
            "composition_only_rate": float(np.mean(only)),
        }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), **summary}, indent=2))


if __name__ == "__main__":
    main()
