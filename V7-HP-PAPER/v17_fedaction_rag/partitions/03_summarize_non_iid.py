#!/usr/bin/env python3
"""Summarize document and query heterogeneity for every frozen partition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def manifests(root: Path) -> list[dict[str, Any]]:
    output = []
    for name in (
        "topic_silo_manifest.json",
        "entity_community_manifest.json",
        "random_control_manifest.json",
        "dirichlet_manifest.json",
    ):
        path = root / name
        if path.exists():
            output.extend(json.loads(path.read_text(encoding="utf-8")).get("datasets", {}).values())
    return output


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    probabilities = [value / total for value in counts if value]
    return -sum(value * math.log(value) for value in probabilities) / math.log(len(counts))


def coefficient_of_variation(counts: list[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    return float(values.std() / values.mean()) if values.mean() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--query-origins", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("non_iid_statistics.md"))
    args = parser.parse_args()

    query_counts: dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    with args.query_origins.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query_counts[(row["dataset"], row["partition"], row["split"])][int(row["origin_client"])] += 1

    lines = [
        "# V17 Non-IID Statistics",
        "",
        "All partitions and query origins are label-free. Dirichlet settings are controlled stress tests, not claims about real organizational silos.",
        "",
        "## Document Distribution",
        "",
        "| Dataset | Partition | M | Documents | Min | Max | Max/mean | CV | Normalized entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    specs = sorted(manifests(args.partition_root), key=lambda row: (row["dataset"], row["partition"]))
    for spec in specs:
        counts = list(map(int, spec["client_counts"]))
        lines.append(
            f"| {spec['dataset']} | {spec['partition']} | {spec['m']} | {sum(counts)} | "
            f"{min(counts)} | {max(counts)} | {max(counts)/(sum(counts)/len(counts)):.3f} | "
            f"{coefficient_of_variation(counts):.3f} | {normalized_entropy(counts):.3f} |"
        )

    lines.extend([
        "",
        "## Query-Origin Distribution",
        "",
        "| Dataset | Partition | Split | Queries | Min | Max | CV | Normalized entropy |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for key in sorted(query_counts):
        counts = [query_counts[key].get(client, 0) for client in range(20)]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {sum(counts)} | {min(counts)} | {max(counts)} | "
            f"{coefficient_of_variation(counts):.3f} | {normalized_entropy(counts):.3f} |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "document_cells": len(specs), "query_cells": len(query_counts)}, indent=2))


if __name__ == "__main__":
    main()
