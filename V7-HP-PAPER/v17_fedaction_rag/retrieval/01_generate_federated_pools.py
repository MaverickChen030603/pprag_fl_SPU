#!/usr/bin/env python3
"""Generate strict client-local retrieval pools under a frozen client budget."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def tokens(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value) if len(token) > 1]


def minmax(values: list[float]) -> list[float]:
    lower, upper = min(values), max(values)
    if upper <= lower:
        return [0.0] * len(values)
    return [(value - lower) / (upper - lower) for value in values]


def fts_query(question: str) -> str:
    selected = list(dict.fromkeys(tokens(question)))[:32]
    return " OR ".join(f'"{value}"' for value in selected)


def sparse_search(connection: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    query = fts_query(question)
    if not query:
        return []
    sql = """
      SELECT d.doc_id, d.title, d.text, -bm25(docs_fts, 2.0, 1.0) AS sparse_score
      FROM docs_fts JOIN docs d ON d.id=docs_fts.rowid
      WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts, 2.0, 1.0) LIMIT ?
    """
    names = ("doc_id", "title", "text", "sparse_score")
    return [dict(zip(names, values)) for values in connection.execute(sql, (query, limit))]


def install_client_assignments(connection: sqlite3.Connection, assignment: dict[str, int]) -> None:
    connection.execute("CREATE TEMP TABLE client_assignment(doc_id TEXT PRIMARY KEY, client_id INTEGER NOT NULL)")
    connection.executemany(
        "INSERT INTO client_assignment(doc_id,client_id) VALUES (?,?)",
        assignment.items(),
    )
    connection.execute("CREATE INDEX temp.client_assignment_client ON client_assignment(client_id)")


def local_sparse_search(
    connection: sqlite3.Connection,
    question: str,
    client_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    query = fts_query(question)
    if not query:
        return []
    sql = """
      SELECT d.doc_id, d.title, d.text, -bm25(docs_fts, 2.0, 1.0) AS sparse_score
      FROM docs_fts
      JOIN docs d ON d.id=docs_fts.rowid
      JOIN client_assignment a ON a.doc_id=d.doc_id
      WHERE docs_fts MATCH ? AND a.client_id=?
      ORDER BY bm25(docs_fts, 2.0, 1.0) LIMIT ?
    """
    names = ("doc_id", "title", "text", "sparse_score")
    output = [dict(zip(names, values)) for values in connection.execute(sql, (query, client_id, limit))]
    for doc in output:
        doc["client_id"] = client_id
    return output


def local_shard_search(connection: sqlite3.Connection, question: str, client_id: int, limit: int) -> list[dict[str, Any]]:
    output = sparse_search(connection, question, limit)
    for doc in output:
        doc["client_id"] = client_id
    return output


def load_assignment(path: Path) -> dict[str, int]:
    return {str(row["doc_id"]): int(row["client_id"]) for row in rows(path)}


def load_origins(path: Path, dataset: str, partition: str, split: str) -> dict[str, int]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["partition"] == partition and row["split"] == split:
                output[str(row["query_id"])] = int(row["origin_client"])
    return output


def bridge_score(question: str, title: str) -> float:
    query_tokens, title_tokens = set(tokens(question)), set(tokens(title))
    return len(query_tokens & title_tokens) / max(1, len(query_tokens | title_tokens))


def encode_query(model, question: str) -> np.ndarray:
    return model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0]


def rank_candidates(
    model,
    question: str,
    candidates: list[dict[str, Any]],
    alpha: float,
    query_embedding: np.ndarray,
) -> list[dict[str, Any]]:
    texts = [f"{doc['title']}. {doc['text']}" for doc in candidates]
    doc_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
    dense = (doc_embeddings @ query_embedding).astype(float).tolist()
    sparse = [float(doc["sparse_score"]) for doc in candidates]
    dense_norm, sparse_norm = minmax(dense), minmax(sparse)
    for index, doc in enumerate(candidates):
        doc["dense_score"] = dense[index]
        doc["sparse_score"] = sparse[index]
        doc["hybrid_score"] = alpha * dense_norm[index] + (1.0 - alpha) * sparse_norm[index]
        doc["retrieval_score"] = doc["hybrid_score"]
        doc["bridge_score"] = bridge_score(question, str(doc["title"]))
    return candidates


def unique_docs(groups: Iterable[Iterable[dict[str, Any]]]) -> list[dict[str, Any]]:
    output, seen = [], set()
    for group in groups:
        for doc in group:
            if doc["doc_id"] not in seen:
                seen.add(doc["doc_id"])
                output.append(doc)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", default="development")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--local-index-root", type=Path)
    parser.add_argument("--centroids", type=Path)
    parser.add_argument("--origins", type=Path)
    parser.add_argument("--centralized", action="store_true")
    parser.add_argument("--client-budget", type=int, default=3)
    parser.add_argument("--local-k", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--sparse-candidates", type=int, default=5000)
    parser.add_argument("--local-sparse-candidates", type=int, default=100)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    if not args.centralized and not all((args.centroids, args.origins)):
        raise ValueError("federated mode requires centroids and origins")
    if not args.centralized and args.local_index_root is None and args.assignment is None:
        raise ValueError("federated mode requires --assignment or --local-index-root")
    model = SentenceTransformer(args.encoder, device=args.device)
    assignment = {} if args.centralized or args.local_index_root else load_assignment(args.assignment)
    centroids = None if args.centralized else np.load(args.centroids)
    origins = {} if args.centralized else load_origins(args.origins, args.dataset, args.partition, args.split_name)
    connection = sqlite3.connect(args.index)
    local_connections: dict[int, sqlite3.Connection] = {}
    if not args.centralized and args.local_index_root is None:
        install_client_assignments(connection, assignment)
    elif not args.centralized:
        for client in range(len(centroids)):
            path = args.local_index_root / f"client_{client:02d}.sqlite"
            if not path.exists():
                raise FileNotFoundError(path)
            local_connections[client] = sqlite3.connect(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count, latencies = 0, []
    with args.output.open("w", encoding="utf-8") as handle:
        try:
            for index, row in enumerate(rows(args.split)):
                if args.max_queries is not None and index >= args.max_queries:
                    break
                started = time.perf_counter()
                query_id, question = qid(row), str(row["question"])
                query_embedding = encode_query(model, question)
                if args.centralized:
                    candidates = sparse_search(connection, question, args.sparse_candidates)
                else:
                    search = local_shard_search if local_connections else local_sparse_search
                    candidates = [
                        doc
                        for client in range(len(centroids))
                        for doc in search(
                            local_connections[client] if local_connections else connection,
                            question,
                            client,
                            args.local_sparse_candidates,
                        )
                    ]
                candidates = rank_candidates(
                    model, question, candidates, args.alpha, query_embedding
                )
                ranked = sorted(candidates, key=lambda doc: (-doc["hybrid_score"], -doc["dense_score"], doc["doc_id"]))
                if args.centralized:
                    action_pool = ranked[:args.pool_size]
                    if len(action_pool) < args.pool_size:
                        raise RuntimeError(f"{query_id}: centralized pool has only {len(action_pool)} documents")
                    for local_rank, doc in enumerate(action_pool):
                        doc.update({"client_id": 0, "local_rank": local_rank})
                    payload = {
                        "query_id": query_id,
                        "dataset": args.dataset,
                        "partition": "centralized",
                        "origin_client": 0,
                        "selected_clients": [0],
                        "client_budget": 1,
                        "local_k": args.pool_size,
                        "pool_size": args.pool_size,
                        "pool": action_pool,
                        "baseline_doc_ids": [doc["doc_id"] for doc in action_pool[:5]],
                        "single_client_contexts": {"0": [doc["doc_id"] for doc in action_pool[:5]]},
                        "router": "centralized_global_hybrid",
                    }
                else:
                    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    for doc in ranked:
                        client = int(doc["client_id"])
                        if len(grouped[client]) < args.local_k:
                            doc["local_rank"] = len(grouped[client])
                            grouped[client].append(doc)
                    missing = [client for client in range(len(centroids)) if len(grouped[client]) < args.local_k]
                    if missing:
                        raise RuntimeError(f"{query_id}: fewer than local-k matches for clients {missing[:5]}; increase sparse-candidates")
                    origin = origins.get(query_id)
                    if origin is None:
                        raise KeyError(f"{query_id}: missing origin for {args.partition}")
                    client_scores = centroids @ query_embedding
                    others = [int(value) for value in np.argsort(-client_scores) if int(value) != origin]
                    selected_clients = [origin] + others[: max(0, args.client_budget - 1)]
                    baseline = list(grouped[origin][:5])
                    additions = sorted(
                        [doc for client in selected_clients[1:] for doc in grouped[client]],
                        key=lambda doc: (-doc["hybrid_score"], doc["client_id"], doc["local_rank"]),
                    )
                    action_pool = unique_docs((baseline, additions))[:args.pool_size]
                    if len(action_pool) < args.pool_size:
                        raise RuntimeError(f"{query_id}: federated action pool has only {len(action_pool)} documents")
                    all_local = unique_docs(grouped[client] for client in range(len(centroids)))
                    extras = [doc for doc in all_local if doc["doc_id"] not in {value["doc_id"] for value in action_pool}]
                    payload = {
                        "query_id": query_id,
                        "dataset": args.dataset,
                        "partition": args.partition,
                        "origin_client": origin,
                        "selected_clients": selected_clients,
                        "client_scores": {str(client): float(client_scores[client]) for client in selected_clients},
                        "client_budget": args.client_budget,
                        "local_k": args.local_k,
                        "pool_size": args.pool_size,
                        "pool": action_pool + extras,
                        "baseline_doc_ids": [doc["doc_id"] for doc in baseline],
                        "single_client_contexts": {str(client): [doc["doc_id"] for doc in grouped[client][:5]] for client in range(len(centroids))},
                        "router": "origin_plus_centroid",
                    }
                elapsed = 1000.0 * (time.perf_counter() - started)
                payload["retrieval_latency_ms"] = elapsed
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                latencies.append(elapsed)
                count += 1
        finally:
            connection.close()
            for local_connection in local_connections.values():
                local_connection.close()
    manifest = {
        "status": "complete",
        "dataset": args.dataset,
        "partition": "centralized" if args.centralized else args.partition,
        "queries": count,
        "client_budget": 1 if args.centralized else args.client_budget,
        "local_k": args.pool_size if args.centralized else args.local_k,
        "context_budget": 5,
        "action_pool_size": args.pool_size,
        "local_sparse_candidates": None if args.centralized else args.local_sparse_candidates,
        "local_indexes": str(args.local_index_root.resolve()) if args.local_index_root else None,
        "encoder": args.encoder,
        "gold_or_support_injection": False,
        "random_padding": False,
        "mean_retrieval_ms": sum(latencies) / len(latencies) if latencies else None,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
