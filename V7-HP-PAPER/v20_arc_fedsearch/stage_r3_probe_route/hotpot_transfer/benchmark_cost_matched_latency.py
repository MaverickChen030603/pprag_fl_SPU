#!/usr/bin/env python3
"""Measure label-free local retrieval service time for frozen R3-C contracts."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_probe_audit import CLIENTS, sparse_search


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--probe-packets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))["profiles"]
    p0 = np.asarray([profile["p0_single_centroid"] for profile in profiles], dtype=np.float32)
    probe_latency = {str(packet["query_id"]): float(packet["probe_materialization_latency_ms"]) for packet in rows(args.probe_packets)}
    if len(probe_latency) != 300:
        raise AssertionError("expected 300 frozen probe packets")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=args.device)
    connections = {client: sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite") for client in range(CLIENTS)}
    per_query = []
    try:
        for index, row in enumerate(rows(args.split), 1):
            question = str(row["question"])
            query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
            ordered = [int(value) for value in np.argsort(-(p0 @ query_embedding), kind="stable")[:8]]
            client_ms = []
            for client in ordered:
                started = time.perf_counter()
                candidates = sparse_search(connections[client], question, 100)
                if candidates:
                    model.encode([f"{item['title']}. {item['text']}" for item in candidates], normalize_embeddings=True,
                                 convert_to_numpy=True, batch_size=256, show_progress_bar=False)
                client_ms.append((time.perf_counter() - started) * 1000.0)
            prefix = np.cumsum(client_ms)
            query = qid(row)
            per_query.extend([
                {"query_id": query, "method": "C0_static_top3", "local_retrieval_latency_ms": float(prefix[2]), "clients_retrieved": 3, "reader_started": False},
                {"query_id": query, "method": "C1_static_top4", "local_retrieval_latency_ms": float(prefix[3]), "clients_retrieved": 4, "reader_started": False},
                {"query_id": query, "method": "C2_static_top8_tail_top1", "local_retrieval_latency_ms": float(prefix[7]), "clients_retrieved": 8, "reader_started": False},
                {"query_id": query, "method": "ProbeRoute_B4_shallow8", "local_retrieval_latency_ms": probe_latency[query], "clients_retrieved": 8, "reader_started": False},
            ])
            if index % 25 == 0:
                print(json.dumps({"completed": index, "target": 300}), flush=True)
    finally:
        for connection in connections.values():
            connection.close()
    summary = []
    for method in ("C0_static_top3", "C1_static_top4", "C2_static_top8_tail_top1", "ProbeRoute_B4_shallow8"):
        values = [item for item in per_query if item["method"] == method]
        summary.append({"method": method, "queries": len(values), "mean_local_retrieval_latency_ms": float(np.mean([item["local_retrieval_latency_ms"] for item in values])),
                        "median_local_retrieval_latency_ms": float(np.median([item["local_retrieval_latency_ms"] for item in values])), "clients_retrieved": values[0]["clients_retrieved"],
                        "measurement_scope": "serial local sparse-plus-dense service compute; network latency excluded", "reader_started": False})
    write_csv(args.output_dir / "latency_per_query.csv", per_query)
    write_csv(args.output_dir / "latency_main_results.csv", summary)
    (args.output_dir / "latency_contract.json").write_text(json.dumps({
        "probe_route_latency_source": "frozen probe packet materialization timing for all 8 P0 candidates",
        "static_latency_source": "fresh label-free serial local sparse-plus-dense replay on identical 300 queries",
        "network_latency_measured": False, "reader_started": False, "final_test_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
