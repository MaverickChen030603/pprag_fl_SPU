#!/usr/bin/env python3
"""Audit C2 communication bytes without accessing labels or metrics."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
M2 = "m2_logistic_proberoute"


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def payload_bytes(connection: sqlite3.Connection, doc_id: str, cache: dict[str, int]) -> int:
    if doc_id not in cache:
        row = connection.execute("SELECT title, text FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(f"Missing transmitted document: {doc_id}")
        cache[doc_id] = len((str(row[0]) + "\n" + str(row[1])).encode("utf-8"))
    return cache[doc_id]


def write_csv(path: Path, values: list[dict]) -> None:
    fields = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()

    retrieval = list(rows(args.stage / "retrieval/blind_retrieval_outputs.jsonl"))
    packets = {
        dataset: {str(row["query_id"]): row for row in rows(args.stage / f"retrieval/{dataset}_probe_packets.jsonl")}
        for dataset in DATASETS
    }
    connections = {
        dataset: sqlite3.connect(f"file:{args.base / f'inputs/indexes/{dataset}.sqlite'}?mode=ro", uri=True)
        for dataset in DATASETS
    }
    caches = {dataset: {} for dataset in DATASETS}
    costs: list[dict] = []
    try:
        for row in retrieval:
            dataset = row["dataset"]
            documents = list(row["transmitted_doc_ids"])
            document_bytes = sum(payload_bytes(connections[dataset], doc_id, caches[dataset]) for doc_id in documents)
            probe_json = 0
            probe_float32 = 0
            if row["method"] == M2:
                # The frozen C2 contract defines eight 18-float packets: 8 * 18 * 4.
                probe_float32 = 8 * 18 * 4
                packet = packets[dataset][str(row["query_id"])]
                probe_json = len(json.dumps(
                    {"query_id": packet["query_id"], "p0_candidate_records": packet["p0_candidate_records"]},
                    separators=(",", ":"), sort_keys=True,
                ).encode("utf-8"))
            costs.append({
                "dataset": dataset,
                "query_id": str(row["query_id"]),
                "method": row["method"],
                "documents": len(documents),
                "document_utf8_payload_bytes": document_bytes,
                "probe_float32_bytes": probe_float32,
                "probe_json_diagnostic_bytes": probe_json,
                "minimum_total_bytes": document_bytes + probe_float32,
                "json_diagnostic_total_bytes": document_bytes + probe_json,
            })
    finally:
        for connection in connections.values():
            connection.close()

    output = args.stage / "statistics"
    output.mkdir(exist_ok=True)
    write_csv(output / "per_query_communication_cost.csv", costs)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in costs:
        grouped[(row["dataset"], row["method"])].append(row)
    summary = []
    for (dataset, method), values in sorted(grouped.items()):
        summary.append({
            "dataset": dataset,
            "method": method,
            "queries": len(values),
            **{key: sum(float(row[key]) for row in values) / len(values) for key in (
                "documents", "document_utf8_payload_bytes", "probe_float32_bytes",
                "probe_json_diagnostic_bytes", "minimum_total_bytes", "json_diagnostic_total_bytes",
            )},
        })
    write_csv(output / "communication_cost_summary.csv", summary)
    (output / "communication_cost_audit.json").write_text(json.dumps({
        "status": "complete",
        "label_accessed": False,
        "document_serialization": "UTF-8(title + newline + text)",
        "m2_probe_float32_bytes_per_query": 576,
        "protocol_overhead": "Not asserted as a wire-level byte count; JSON diagnostic is reported separately.",
        "rows": len(costs),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
