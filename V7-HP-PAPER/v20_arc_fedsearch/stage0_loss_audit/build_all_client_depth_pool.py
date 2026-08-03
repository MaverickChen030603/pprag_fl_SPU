#!/usr/bin/env python3
"""Materialize an all-client local-depth pool without using evaluation labels.

The inherited V17 pool exposes only local-k=5.  This builder preserves the
frozen V17 selected-client set for each query, but independently asks every
physical local shard for a deeper local list.  It is therefore suitable for a
local-depth and document-budget audit, not for claiming a new router.
"""

from __future__ import annotations

import argparse
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


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def fts_query(question: str) -> str:
    terms = list(dict.fromkeys(token.lower() for token in TOKEN_RE.findall(question) if len(token) > 1))[:32]
    return " OR ".join(f'"{term}"' for term in terms)


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
    return [dict(zip(names, value)) for value in connection.execute(sql, (query, limit))]


def minmax(values: list[float]) -> list[float]:
    lower, upper = min(values), max(values)
    if upper <= lower:
        return [0.0] * len(values)
    return [(value - lower) / (upper - lower) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--inherited-pool", type=Path, required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--local-depth", type=int, default=10)
    parser.add_argument("--sparse-candidates", type=int, default=100)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    inherited = {str(row["query_id"]): row for row in rows(args.inherited_pool)}
    source = list(rows(args.split))
    if args.max_queries is not None:
        source = source[: args.max_queries]
    source = [row for row in source if query_id(row) in inherited]
    if not source:
        raise ValueError("no development queries align with the inherited pool")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and args.output.exists():
        completed = {str(row["query_id"]) for row in rows(args.output)}
    elif args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")

    model = SentenceTransformer(args.encoder, device=args.device)
    connections = {
        client: sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite")
        for client in range(args.clients)
    }
    started = time.perf_counter()
    emitted = 0
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(source, start=1):
                qid = query_id(row)
                if qid in completed:
                    continue
                question = str(row["question"])
                query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
                candidates: dict[int, list[dict[str, Any]]] = {}
                flat: list[dict[str, Any]] = []
                for client, connection in connections.items():
                    values = sparse_search(connection, question, args.sparse_candidates)
                    for value in values:
                        value["client_id"] = client
                    candidates[client] = values
                    flat.extend(values)
                texts = [f"{doc['title']}. {doc['text']}" for doc in flat]
                embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=args.batch_size, show_progress_bar=False)
                dense = (embeddings @ query_embedding).astype(float).tolist()
                for doc, score in zip(flat, dense):
                    doc["dense_score"] = score
                local_lists: dict[int, list[dict[str, Any]]] = {}
                for client, values in candidates.items():
                    if not values:
                        local_lists[client] = []
                        continue
                    dense_norm = minmax([float(doc["dense_score"]) for doc in values])
                    sparse_norm = minmax([float(doc["sparse_score"]) for doc in values])
                    for doc, dense_value, sparse_value in zip(values, dense_norm, sparse_norm):
                        doc["local_dense_norm"] = dense_value
                        doc["local_sparse_norm"] = sparse_value
                        doc["local_hybrid_score"] = 0.55 * dense_value + 0.45 * sparse_value
                        # The inherited replay expects this field for its raw local-score merger.
                        doc["hybrid_score"] = doc["local_hybrid_score"]
                    ranked = sorted(values, key=lambda doc: (-float(doc["local_hybrid_score"]), -float(doc["dense_score"]), str(doc["doc_id"])))[: args.local_depth]
                    for rank, doc in enumerate(ranked):
                        doc["local_rank"] = rank
                    local_lists[client] = ranked
                payload = {
                    "query_id": qid,
                    "dataset": args.dataset,
                    "selected_clients": inherited[qid]["selected_clients"],
                    "inherited_router": inherited[qid].get("router", "unknown"),
                    "local_depth": args.local_depth,
                    "sparse_candidates_per_client": args.sparse_candidates,
                    "score_contract": "client-local minmax dense/sparse hybrid; no gold fields",
                    "pool": [doc for client in range(args.clients) for doc in local_lists[client]],
                    "local_candidate_counts": {str(client): len(values) for client, values in candidates.items()},
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                emitted += 1
                if emitted % 10 == 0:
                    elapsed = time.perf_counter() - started
                    print(json.dumps({"status": "running", "completed": emitted + len(completed), "target": len(source), "elapsed_s": round(elapsed, 1)}), flush=True)
    finally:
        for connection in connections.values():
            connection.close()
    manifest = {
        "status": "complete",
        "dataset": args.dataset,
        "queries": len(completed) + emitted,
        "local_depth": args.local_depth,
        "sparse_candidates_per_client": args.sparse_candidates,
        "encoder": args.encoder,
        "gold_or_answer_fields_used": False,
        "inherited_router_only": True,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
