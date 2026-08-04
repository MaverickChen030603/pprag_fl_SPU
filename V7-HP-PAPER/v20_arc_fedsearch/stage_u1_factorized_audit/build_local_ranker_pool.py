#!/usr/bin/env python3
"""Materialize frozen dense/BM25/hybrid/RRF local rankings without labels."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TOKEN = re.compile(r"[A-Za-z0-9]+")
RANKERS = ("L0_dense", "L1_bm25", "L2_hybrid", "L3_rrf")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def fts_query(question: str) -> str:
    terms = list(dict.fromkeys(token.lower() for token in TOKEN.findall(question) if len(token) > 1))[:32]
    return " OR ".join(f'"{term}"' for term in terms)


def sparse_search(connection: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    query = fts_query(question)
    if not query:
        return []
    sql = """SELECT d.doc_id, d.title, d.text, -bm25(docs_fts,2.0,1.0) AS sparse_score
             FROM docs_fts JOIN docs d ON d.id=docs_fts.rowid
             WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts,2.0,1.0) LIMIT ?"""
    names = ("doc_id", "title", "text", "sparse_score")
    return [dict(zip(names, value)) for value in connection.execute(sql, (query, limit))]


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.0] * len(values) if hi <= lo else [(value - lo) / (hi - lo) for value in values]


def compact(doc: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    return {"doc_id": str(doc["doc_id"]), "title": str(doc["title"]), "client_id": int(doc["client_id"]),
            "local_rank": rank, "local_score": float(score), "dense_score": float(doc["dense_score"]),
            "sparse_score": float(doc["sparse_score"]), "hybrid_score": float(doc["hybrid_score"]),
            "rrf_score": float(doc["rrf_score"])}


def rank_local(values: list[dict[str, Any]], depth: int) -> dict[str, list[dict[str, Any]]]:
    dense = sorted(values, key=lambda d: (-float(d["dense_score"]), str(d["doc_id"])))
    bm25 = sorted(values, key=lambda d: (-float(d["sparse_score"]), str(d["doc_id"])))
    dense_rank = {str(doc["doc_id"]): rank for rank, doc in enumerate(dense)}
    bm25_rank = {str(doc["doc_id"]): rank for rank, doc in enumerate(bm25)}
    for doc in values:
        doc["rrf_score"] = 1.0 / (61 + dense_rank[str(doc["doc_id"])]) + 1.0 / (61 + bm25_rank[str(doc["doc_id"])])
    rankings = {
        "L0_dense": dense,
        "L1_bm25": bm25,
        "L2_hybrid": sorted(values, key=lambda d: (-float(d["hybrid_score"]), -float(d["dense_score"]), str(d["doc_id"]))),
        "L3_rrf": sorted(values, key=lambda d: (-float(d["rrf_score"]), -float(d["dense_score"]), str(d["doc_id"]))),
    }
    score_key = {"L0_dense": "dense_score", "L1_bm25": "sparse_score", "L2_hybrid": "hybrid_score", "L3_rrf": "rrf_score"}
    return {
        name: [compact(doc, rank, float(doc[score_key[name]])) for rank, doc in enumerate(ranked[:depth])]
        for name, ranked in rankings.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-depth", type=int, default=10)
    parser.add_argument("--sparse-candidates", type=int, default=100)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    from sentence_transformers import SentenceTransformer
    source = list(rows(args.split))
    completed = {str(row["query_id"]) for row in rows(args.output)} if args.resume and args.output.exists() else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.encoder, device=args.device)
    connections = {client: sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite") for client in range(args.clients)}
    started, emitted = time.perf_counter(), 0
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for row in source:
                query = qid(row)
                if query in completed:
                    continue
                question = str(row["question"])
                query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
                by_client, flat = {}, []
                for client, connection in connections.items():
                    values = sparse_search(connection, question, args.sparse_candidates)
                    for doc in values:
                        doc["client_id"] = client
                    by_client[client] = values
                    flat.extend(values)
                embeddings = model.encode([f"{doc['title']}. {doc['text']}" for doc in flat], normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
                for doc, score in zip(flat, (embeddings @ query_embedding).astype(float).tolist()):
                    doc["dense_score"] = score
                rankings = {}
                for client, values in by_client.items():
                    dn, sn = minmax([float(d["dense_score"]) for d in values]), minmax([float(d["sparse_score"]) for d in values])
                    for doc, dscore, sscore in zip(values, dn, sn):
                        doc["hybrid_score"] = 0.55 * dscore + 0.45 * sscore
                    rankings[str(client)] = rank_local(values, args.local_depth)
                handle.write(json.dumps({"query_id": query, "dataset": args.dataset, "local_depth": args.local_depth,
                                         "rankers": rankings, "candidate_count_per_client": {str(k): len(v) for k, v in by_client.items()},
                                         "gold_or_answer_fields_used": False}, ensure_ascii=False) + "\n")
                handle.flush(); emitted += 1
                if emitted % 10 == 0:
                    print(json.dumps({"status": "running", "completed": emitted + len(completed), "target": len(source), "elapsed_s": round(time.perf_counter()-started, 1)}), flush=True)
    finally:
        for connection in connections.values(): connection.close()
    manifest = {"status": "complete", "dataset": args.dataset, "queries": emitted + len(completed), "rankers": RANKERS,
                "local_depth": args.local_depth, "sparse_candidates": args.sparse_candidates, "gold_or_answer_fields_used": False,
                "reader_started": False, "output": str(args.output.resolve())}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
