#!/usr/bin/env python3
"""Validate total, unique, label-free ownership across frozen client partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def specs(root: Path) -> list[dict[str, Any]]:
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


def index_ids(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {str(row[0]) for row in connection.execute("SELECT doc_id FROM docs")}
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("partition_audit.json"))
    args = parser.parse_args()
    failures, warnings, cells = [], [], []
    expected_partitions = {
        "topic_silo", "entity_community", "random_control",
        "dirichlet_a0p1", "dirichlet_a0p3", "dirichlet_a1p0",
    }
    all_specs = specs(args.partition_root)
    observed = {(spec["dataset"], spec["partition"]) for spec in all_specs}
    expected = {
        (dataset, partition)
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
        for partition in expected_partitions
    }
    for missing in sorted(expected - observed):
        failures.append(f"missing manifest cell: {missing[0]}/{missing[1]}")

    cached_indexes: dict[str, set[str]] = {}
    for spec in sorted(all_specs, key=lambda row: (row["dataset"], row["partition"])):
        dataset, partition, m = spec["dataset"], spec["partition"], int(spec["m"])
        assignment_path = Path(spec["assignment_path"])
        assignment_rows = list(rows(assignment_path))
        ids = [str(row["doc_id"]) for row in assignment_rows]
        clients = [int(row["client_id"]) for row in assignment_rows]
        forbidden = [key for row in assignment_rows for key in row if key not in {"doc_id", "client_id"}]
        if forbidden:
            failures.append(f"{dataset}/{partition}: unexpected assignment fields {sorted(set(forbidden))}")
        if len(ids) != len(set(ids)):
            failures.append(f"{dataset}/{partition}: duplicate document ownership")
        if any(client < 0 or client >= m for client in clients):
            failures.append(f"{dataset}/{partition}: client ID outside [0,{m})")
        if sha256(assignment_path) != spec["assignment_sha256"]:
            failures.append(f"{dataset}/{partition}: assignment hash mismatch")
        source = str(Path(spec["source_index"]).resolve())
        if source not in cached_indexes:
            cached_indexes[source] = index_ids(Path(source))
        source_ids = cached_indexes[source]
        if set(ids) != source_ids:
            failures.append(
                f"{dataset}/{partition}: assignment/index mismatch "
                f"missing={len(source_ids-set(ids))} extra={len(set(ids)-source_ids)}"
            )
        counts = [clients.count(client) for client in range(m)]
        if min(counts) == 0:
            failures.append(f"{dataset}/{partition}: empty client")
        ratio = max(counts) / (sum(counts) / len(counts))
        if partition == "random_control" and max(counts) - min(counts) > 1:
            failures.append(f"{dataset}/{partition}: random control is not balanced")
        if partition in {"topic_silo", "entity_community"} and ratio > 5.0:
            failures.append(f"{dataset}/{partition}: max/mean imbalance {ratio:.3f} exceeds 5.0")
        elif partition in {"topic_silo", "entity_community"} and ratio > 3.0:
            warnings.append(f"{dataset}/{partition}: max/mean imbalance is {ratio:.3f}")
        centroids = np.load(spec["centroid_path"])
        if centroids.shape[0] != m or not np.isfinite(centroids).all():
            failures.append(f"{dataset}/{partition}: invalid centroid array {centroids.shape}")
        cells.append({
            "dataset": dataset,
            "partition": partition,
            "documents": len(ids),
            "clients": m,
            "min_client_size": min(counts),
            "max_client_size": max(counts),
            "max_to_mean_ratio": ratio,
            "gold_labels_used": bool(spec.get("gold_labels_used", True)),
        })
        if spec.get("gold_labels_used", True):
            failures.append(f"{dataset}/{partition}: manifest does not certify label-free construction")
    payload = {
        "status": "pass" if not failures else "fail",
        "expected_cells": len(expected),
        "observed_cells": len(observed),
        "cells": cells,
        "warnings": warnings,
        "failures": failures,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
