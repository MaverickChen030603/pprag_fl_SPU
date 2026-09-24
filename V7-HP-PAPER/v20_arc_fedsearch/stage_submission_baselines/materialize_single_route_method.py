#!/usr/bin/env python3
"""Materialize one Top-8/Top-3 route method from frozen unlabeled R5 packets."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from materialize_posthoc_baselines import (
    context_hash,
    lookup_documents,
    rows,
    sha256,
)


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def raw_merge_variable(packet: dict[str, Any], clients: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transmitted = [
        dict(document)
        for client in clients
        for document in packet["local_dense_docs_top10"][str(client)][:5]
    ]
    expected = 5 * len(clients)
    if len(transmitted) != expected:
        raise ValueError(f"expected {expected} transmitted documents, found {len(transmitted)}")
    merged = sorted(
        transmitted,
        key=lambda document: (-float(document["dense_score"]), str(document["doc_id"])),
    )[:10]
    return transmitted, merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--client-budget", type=int, default=3)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "contexts_unlabeled.jsonl"
    summaries = {}
    with output.open("x", encoding="utf-8") as handle:
        for dataset in DATASETS:
            split = {
                str(row.get("query_id", row.get("_id", row.get("id")))): row
                for row in rows(
                    args.input_root
                    / "protocol"
                    / f"{dataset}_final_test_inputs_n300.jsonl"
                )
            }
            packets = {
                str(row["query_id"]): row
                for row in rows(args.input_root / "retrieval" / f"{dataset}_probe_packets.jsonl")
            }
            route_path = args.route_root / dataset / "r5_routes_unlabeled.jsonl"
            routes = {str(row["query_id"]): row for row in rows(route_path)}
            if set(split) != set(packets) or set(split) != set(routes):
                raise ValueError(f"{dataset}: sample, packets, and routes must match")
            if {str(row["method"]) for row in routes.values()} != {args.method}:
                raise ValueError(f"{dataset}: route method does not match {args.method}")
            connection = sqlite3.connect(
                f"file:{args.index_root / (dataset + '.sqlite')}?mode=ro", uri=True
            )
            try:
                for qid, source in split.items():
                    route = routes[qid]
                    clients = [int(value) for value in route["selected_clients"]]
                    candidates = {
                        int(item["client_id"])
                        for item in packets[qid]["p0_candidate_records"]
                    }
                    if args.client_budget > 0:
                        if len(clients) != args.client_budget or not set(clients).issubset(candidates):
                            raise ValueError(f"{dataset}/{qid}: route violates Top-8/Top-{args.client_budget}")
                    elif sorted(clients) != list(range(20)):
                        raise ValueError(f"{dataset}/{qid}: all-client route must select clients 0..19")
                    transmitted, merged = raw_merge_variable(packets[qid], clients)
                    documents = lookup_documents(
                        connection, [str(item["doc_id"]) for item in merged[:5]]
                    )
                    question = str(source["question"])
                    handle.write(
                        json.dumps(
                            {
                                "dataset": dataset,
                                "query_id": qid,
                                "question": question,
                                "method": args.method,
                                "selected_clients": clients,
                                "transmitted_doc_ids": [
                                    str(item["doc_id"]) for item in transmitted
                                ],
                                "retrieved_doc_ids": [
                                    str(item["doc_id"]) for item in merged
                                ],
                                "reader_context_doc_ids": [
                                    str(item["doc_id"]) for item in merged[:5]
                                ],
                                "reader_context_docs": documents,
                                "context_hash": context_hash(question, documents),
                                "candidate_clients": 8,
                                "client_budget": len(clients),
                                "local_depth": 10,
                                "documents_per_client": 5,
                                "transmission_budget": 5 * len(clients),
                                "global_pool_size": 10,
                                "reader_context_k": 5,
                                "probe_bytes": 0,
                                "gold_or_answer_used": False,
                                "reader_started": False,
                                "evidence_role": "retrospective_post_hoc_r5_revealed",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            finally:
                connection.close()
            summaries[dataset] = {"queries": len(split), "routes_sha256": sha256(route_path)}
    row_count = sum(1 for _ in rows(output))
    if row_count != 900:
        raise ValueError(f"expected 900 contexts, found {row_count}")
    manifest = {
        "status": "complete_unlabeled_contexts",
        "method": args.method,
        "rows": row_count,
        "output_sha256": sha256(output),
        "datasets": summaries,
        "r5_labels_opened": False,
        "reader_started": False,
        "evidence_role": "retrospective_post_hoc_r5_revealed",
    }
    atomic_json(args.output_dir / "materialization_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
