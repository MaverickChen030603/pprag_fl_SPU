#!/usr/bin/env python3
"""Aggregate frozen pool recall, redundancy, and latency statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.statistics:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                groups[int(row["pool_size_target"])].append(row)
    lines = ["# V15 Retrieval Pool Recall", "", "Pools are produced by the frozen BM25/BGE hybrid retriever over real train-derived corpora. No random or gold padding is used.", "", "| Target pool | Queries | Mean actual size | Support recall | Complete support | Answer document | Redundancy | Mean latency (ms) | P95 latency (ms) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for size, rows in sorted(groups.items()):
        numeric = lambda key: [float(row[key]) for row in rows if row.get(key, "") != ""]
        latency = numeric("retrieval_latency_ms")
        record = {
            "queries": len(rows),
            "mean_actual_size": float(np.mean(numeric("pool_size_actual"))),
            "support_recall": float(np.mean(numeric("support_document_recall"))) if numeric("support_document_recall") else None,
            "complete_support": float(np.mean(numeric("all_support_documents_present"))) if numeric("all_support_documents_present") else None,
            "answer_document": float(np.mean(numeric("answer_document_present"))) if numeric("answer_document_present") else None,
            "redundancy": float(np.mean(numeric("mean_pair_redundancy"))),
            "latency_mean_ms": float(np.mean(latency)),
            "latency_p95_ms": float(np.quantile(latency, 0.95)),
        }
        summary[str(size)] = record
        show = lambda value: "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {size} | {record['queries']} | {record['mean_actual_size']:.2f} | {show(record['support_recall'])} | {show(record['complete_support'])} | {show(record['answer_document'])} | {record['redundancy']:.4f} | {record['latency_mean_ms']:.1f} | {record['latency_p95_ms']:.1f} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

