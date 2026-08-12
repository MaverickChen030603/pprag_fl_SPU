#!/usr/bin/env python3
"""Materialize frozen R5 contexts from unlabeled final queries and probe packets."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_SCHEMA = (
    "dense_top1_score", "dense_top3_mean", "dense_top1_top2_margin", "dense_score_std",
    "dense_score_entropy", "dense_local_rank_percentile", "bm25_top1_score", "bm25_top3_mean",
    "bm25_top1_top2_margin", "dense_bm25_top1_same", "dense_bm25_top3_overlap",
    "dense_sparse_rank_correlation", "matched_query_entity_count", "matched_query_token_count",
    "matched_title_token_count", "query_title_embedding_similarity", "top3_title_diversity", "top3_entity_diversity",
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(values: list[float]) -> list[int]:
    return [int(index) for index in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")]


def minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(array.max() - array.min())
    return np.zeros_like(array) if span <= 1e-12 else (array - array.min()) / span


def label_free(records: list[dict[str, Any]], dataset: str) -> list[int]:
    if dataset == "musique":
        order = stable_rank([float(record["dense_top1_score"]) for record in records])
    else:
        score = .25 * minmax([float(record["static_score"]) for record in records])
        score += .75 * minmax([float(record["dense_top3_mean"]) for record in records])
        order = stable_rank(score.tolist())
    return [int(records[index]["client_id"]) for index in order[:3]]


def logistic(records: list[dict[str, Any]], model_path: Path) -> list[int]:
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    features = np.asarray([[float(record["static_score"]), *[float(record[name]) for name in FEATURE_SCHEMA]] for record in records])
    probability = payload["model"].predict_proba(payload["scaler"].transform(features))[:, 1]
    return [int(records[index]["client_id"]) for index in stable_rank(probability.tolist())[:3]]


def raw_merge(packet: dict[str, Any], clients: list[int]):
    transmitted = [doc for client in clients for doc in packet["local_dense_docs_top10"][str(client)][:5]]
    merged = sorted(transmitted, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:10]
    return transmitted, merged


def lookup(connection: sqlite3.Connection, ids: list[str]) -> list[dict[str, str]]:
    output = []
    for doc_id in ids:
        value = connection.execute("SELECT doc_id,title,text FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        if value is None:
            raise KeyError(f"missing canonical document {doc_id}")
        output.append({"doc_id": str(value[0]), "title": str(value[1]), "text": str(value[2])})
    return output


def context_hash(question: str, docs: list[dict[str, str]]) -> str:
    payload = json.dumps({"question": question, "docs": docs}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--central-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split = {qid(row): row for row in rows(args.split)}
    packets = {str(row["query_id"]): row for row in rows(args.packets)}
    central = {str(row["query_id"]): row for row in rows(args.central_pool)}
    if not (set(split) == set(packets) == set(central)) or len(split) != 300:
        raise ValueError(f"query mismatch split={len(split)} packets={len(packets)} central={len(central)}")
    connection = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    output = []
    try:
        for query_id in [qid(row) for row in rows(args.split)]:
            packet, question = packets[query_id], str(split[query_id]["question"])
            records = packet["p0_candidate_records"]
            static = [int(row["client_id"]) for row in sorted(records, key=lambda row: int(row["static_candidate_rank"]))[:3]]
            inherited = packet.get("inherited_selected_clients")
            methods = {
                "federated_baseline": [int(value) for value in inherited] if args.dataset == "hotpotqa" else static,
                "label_free_proberoute": label_free(records, args.dataset),
                "logistic_proberoute": logistic(records, args.model),
            }
            for method, clients in methods.items():
                transmitted, merged = raw_merge(packet, clients)
                merged_ids = [str(doc["doc_id"]) for doc in merged]
                docs = lookup(connection, merged_ids[:5])
                output.append({"dataset": args.dataset, "query_id": query_id, "question": question, "method": method,
                               "selected_clients": clients, "transmitted_doc_ids": [str(doc["doc_id"]) for doc in transmitted],
                               "retrieved_doc_ids": merged_ids, "reader_context_doc_ids": merged_ids[:5], "reader_context_docs": docs,
                               "context_hash": context_hash(question, docs), "client_budget": 3, "local_depth": 10,
                               "transmission_budget": 15, "global_pool_size": 10, "reader_context_k": 5,
                               "probe_bytes": 592 if method != "federated_baseline" else 0,
                               "gold_or_answer_used": False, "reader_started": False})
            pool = central[query_id].get("pool", [])[:10]
            docs = [{"doc_id": str(doc["doc_id"]), "title": str(doc["title"]), "text": str(doc["text"])} for doc in pool[:5]]
            ids = [str(doc["doc_id"]) for doc in pool]
            output.append({"dataset": args.dataset, "query_id": query_id, "question": question,
                           "method": "centralized_retrieval_reference", "selected_clients": [0],
                           "transmitted_doc_ids": ids, "retrieved_doc_ids": ids, "reader_context_doc_ids": ids[:5],
                           "reader_context_docs": docs, "context_hash": context_hash(question, docs), "client_budget": 1,
                           "local_depth": 10, "transmission_budget": 10, "global_pool_size": 10, "reader_context_k": 5,
                           "probe_bytes": 0, "gold_or_answer_used": False, "reader_started": False})
    finally:
        connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output.with_suffix(".manifest.json").write_text(json.dumps({"status": "complete", "dataset": args.dataset, "queries": 300, "rows": len(output), "methods": 4, "labels_used": False, "split_sha256": sha256(args.split), "packets_sha256": sha256(args.packets), "model_sha256": sha256(args.model), "index_sha256": sha256(args.index), "central_sha256": sha256(args.central_pool), "output_sha256": sha256(args.output)}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "dataset": args.dataset, "rows": len(output), "labels_used": False}, indent=2))


if __name__ == "__main__":
    main()
