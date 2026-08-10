#!/usr/bin/env python3
"""Evaluate frozen R3-C static query-more-client cost baselines from packets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_probe_audit import support_docs


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


def docs(packet: dict[str, Any], limits: dict[int, int]) -> list[dict[str, Any]]:
    pool = packet["local_dense_docs_top10"]
    return [document for client, limit in limits.items() for document in pool[str(client)][:limit]]


def rank_merge(documents: list[dict[str, Any]], merge: str) -> list[dict[str, Any]]:
    if merge == "raw":
        return sorted(documents, key=lambda item: (-float(item["dense_score"]), str(item["doc_id"])))[:10]
    return sorted(documents, key=lambda item: (int(item["local_rank"]), -float(item["dense_score"]), str(item["doc_id"])))[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment)}
    split = {qid(row): row for row in rows(args.split)}
    per_query = []
    for packet in rows(args.packets):
        query = str(packet["query_id"])
        ordered = [int(record["client_id"]) for record in sorted(packet["p0_candidate_records"], key=lambda item: int(item["static_candidate_rank"]))]
        # C0: the original static three-client, 15-document contract.
        # C1: query one more deep client and transmit five more documents.
        # C2: retain C0, but query the remaining P0 candidates and return one
        # document from each; it explicitly exposes the query-more-client cost.
        contracts = {
            "C0_static_top3": ({client: 5 for client in ordered[:3]}, 0, 3),
            "C1_static_top4": ({client: 5 for client in ordered[:4]}, 0, 4),
            "C2_static_top8_tail_top1": ({**{client: 5 for client in ordered[:3]}, **{client: 1 for client in ordered[3:8]}}, 0, 8),
        }
        # All client/document choices above are label-free. Read the sealed
        # labels only after candidates and transmission contracts are fixed.
        gold_docs = support_docs(split[query], "hotpotqa")
        gold_clients = {assignment[document] for document in gold_docs if document in assignment}
        for method, (limits, probe_bytes, deep_clients) in contracts.items():
            transmitted = docs(packet, limits)
            available = {str(document["doc_id"]) for client in limits for document in packet["local_dense_docs_top10"][str(client)][:10]}
            row = {
                "query_id": query, "method": method, "client_budget": len(limits),
                "complete_client_set_recall": int(bool(gold_clients) and gold_clients.issubset(set(limits))),
                "gold_client_recall": len(gold_clients & set(limits)) / max(1, len(gold_clients)),
                "local_complete_at_10": int(bool(gold_docs) and gold_docs.issubset(available)),
                "transmitted_complete": int(bool(gold_docs) and gold_docs.issubset({str(document["doc_id"]) for document in transmitted})),
                "raw_merged_complete_at_10": int(bool(gold_docs) and gold_docs.issubset({str(document["doc_id"]) for document in rank_merge(transmitted, "raw")})),
                "percentile_merged_complete_at_10": int(bool(gold_docs) and gold_docs.issubset({str(document["doc_id"]) for document in rank_merge(transmitted, "percentile")})),
                "shallow_client_compute": 0, "deep_client_compute": deep_clients,
                "probe_bytes": probe_bytes, "document_bytes": sum(int(document["payload_bytes"]) for document in transmitted),
                "total_bytes": probe_bytes + sum(int(document["payload_bytes"]) for document in transmitted),
                "documents_transmitted": len(transmitted), "reader_started": False, "final_test_accessed": False,
            }
            per_query.append(row)
    summary = []
    metrics = ("complete_client_set_recall", "gold_client_recall", "local_complete_at_10", "transmitted_complete", "raw_merged_complete_at_10", "percentile_merged_complete_at_10", "probe_bytes", "document_bytes", "total_bytes", "shallow_client_compute", "deep_client_compute", "documents_transmitted")
    for method in sorted({row["method"] for row in per_query}):
        values = [row for row in per_query if row["method"] == method]
        summary.append({"method": method, "queries": len(values), **{metric: float(np.mean([float(row[metric]) for row in values])) for metric in metrics}, "reader_started": False})
    write_csv(args.output_dir / "cost_per_query.csv", per_query)
    write_csv(args.output_dir / "cost_main_results.csv", summary)
    (args.output_dir / "cost_contract.json").write_text(json.dumps({
        "C0": "static Top-3, 5 documents/client", "C1": "static Top-4, 5 documents/client",
        "C2": "static Top-3 x5 plus P0 ranks 4-8 x1; 8 deep clients, 20 documents",
        "probe_route": "P0 Top-8 shallow probe, 592 metadata bytes, 3 deep clients, 15 documents",
        "reader_started": False, "final_test_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
