#!/usr/bin/env python3
"""Materialize the two frozen R5-C1 contexts from label-free inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np


FEATURE_SCHEMA = (
    "dense_top1_score", "dense_top3_mean", "dense_top1_top2_margin", "dense_score_std",
    "dense_score_entropy", "dense_local_rank_percentile", "bm25_top1_score", "bm25_top3_mean",
    "bm25_top1_top2_margin", "dense_bm25_top1_same", "dense_bm25_top3_overlap",
    "dense_sparse_rank_correlation", "matched_query_entity_count", "matched_query_token_count",
    "matched_title_token_count", "query_title_embedding_similarity", "top3_title_diversity", "top3_entity_diversity",
)
EXPECTED = 4200


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(values) -> list[int]:
    return [int(value) for value in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split_rows = list(rows(args.split))
    if len(split_rows) != EXPECTED or any(set(row) != {"query_id", "question"} for row in split_rows):
        raise ValueError("invalid label-free split")
    split = {str(row["query_id"]): row for row in split_rows}
    packets = {str(row["query_id"]): row for row in rows(args.packets)}
    routes = {str(row["query_id"]): row for row in rows(args.routes)}
    if not (set(split) == set(packets) == set(routes)):
        raise ValueError("query-set mismatch")
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    connection = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    emitted = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for query_id in [str(row["query_id"]) for row in split_rows]:
                packet = packets[query_id]
                records = packet["p0_candidate_records"]
                features = np.asarray(
                    [[float(row["static_score"]), *[float(row[name]) for name in FEATURE_SCHEMA]] for row in records]
                )
                probability = model["model"].predict_proba(model["scaler"].transform(features))[:, 1]
                logistic_clients = [int(records[index]["client_id"]) for index in stable_rank(probability)[:3]]
                methods = {
                    "federated_baseline": [int(value) for value in routes[query_id]["selected_clients"]],
                    "logistic_proberoute": logistic_clients,
                }
                for method, clients in methods.items():
                    transmitted = [
                        doc for client in clients for doc in packet["local_dense_docs_top10"][str(client)][:5]
                    ]
                    merged = sorted(transmitted, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:10]
                    docs = []
                    for doc in merged[:5]:
                        value = connection.execute(
                            "SELECT doc_id,title,text FROM docs WHERE doc_id=?", (str(doc["doc_id"]),)
                        ).fetchone()
                        if value is None:
                            raise KeyError(f"missing document {doc['doc_id']}")
                        docs.append({"doc_id": str(value[0]), "title": str(value[1]), "text": str(value[2])})
                    payload = {
                        "dataset": "hotpotqa",
                        "query_id": query_id,
                        "question": str(split[query_id]["question"]),
                        "method": method,
                        "selected_clients": clients,
                        "reader_context_docs": docs,
                        "client_budget": 3,
                        "local_depth": 10,
                        "documents_per_client": 5,
                        "transmitted_documents": 15,
                        "global_pool": 10,
                        "reader_context_k": 5,
                        "raw_dense_merge": True,
                        "gold_or_answer_used": False,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                    emitted += 1
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        connection.close()
    os.replace(temporary, args.output)
    manifest = {
        "status": "complete",
        "queries": EXPECTED,
        "methods": ["federated_baseline", "logistic_proberoute"],
        "rows": emitted,
        "split_sha256": sha256(args.split),
        "packets_sha256": sha256(args.packets),
        "routes_sha256": sha256(args.routes),
        "model_sha256": sha256(args.model),
        "output_sha256": sha256(args.output),
        "labels_used": False,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    import os

    main()
