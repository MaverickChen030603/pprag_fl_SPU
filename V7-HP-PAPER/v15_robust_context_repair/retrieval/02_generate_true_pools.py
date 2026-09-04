#!/usr/bin/env python3
"""Generate real hybrid Top-10/Top-20 pools from a frozen corpus index."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_common import documents, iter_rows, lexical_tokens, minmax, normalize_title, query_id, support_titles


def fts_query(question: str) -> str:
    tokens = list(dict.fromkeys(lexical_tokens(question)))[:32]
    return " OR ".join(f'"{token}"' for token in tokens)


def search(connection: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    query = fts_query(question)
    if not query:
        return []
    sql = """
        SELECT d.doc_id, d.title, d.text, -bm25(docs_fts, 2.0, 1.0) AS sparse_score
        FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid
        WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts, 2.0, 1.0) LIMIT ?
    """
    return [dict(zip(("doc_id", "title", "text", "sparse_score"), row)) for row in connection.execute(sql, (query, limit))]


def local_sparse(question: str, title: str, text: str) -> float:
    query = set(lexical_tokens(question))
    title_tokens = set(lexical_tokens(title))
    body = set(lexical_tokens(text))
    return 2.0 * len(query & title_tokens) + len(query & body) / math.sqrt(max(1, len(body)))


def jaccard(left: str, right: str) -> float:
    a, b = set(lexical_tokens(left)), set(lexical_tokens(right))
    return len(a & b) / len(a | b) if a and b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--sparse-candidates", type=int, default=200)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.encoder, device=args.device)
    connection = sqlite3.connect(args.index)
    output_paths = {size: args.output_dir / f"top{size}.jsonl" for size in (10, 20)}
    handles = {size: path.open("w", encoding="utf-8") for size, path in output_paths.items()}
    statistics = []
    try:
        for offset, row in enumerate(iter_rows(args.split)):
            if args.max_queries and offset >= args.max_queries:
                break
            started = time.perf_counter()
            question = str(row["question"])
            global_docs = search(connection, question, args.sparse_candidates)
            local_docs = documents(row, args.dataset)
            by_id = {doc["doc_id"]: dict(doc) for doc in global_docs}
            for doc in local_docs:
                existing = by_id.get(doc["doc_id"], {})
                merged = dict(doc)
                merged["sparse_score"] = float(existing.get("sparse_score", local_sparse(question, doc["title"], doc["text"])))
                by_id[doc["doc_id"]] = merged
            candidates = list(by_id.values())
            texts = [f"{doc['title']}. {doc['text']}" for doc in candidates]
            query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            doc_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
            dense = (doc_embeddings @ query_embedding).astype(float).tolist()
            sparse = [float(doc.get("sparse_score", 0.0)) for doc in candidates]
            dense_normalized, sparse_normalized = minmax(dense), minmax(sparse)
            for index, doc in enumerate(candidates):
                doc["dense_score"] = dense[index]
                doc["sparse_score"] = sparse[index]
                doc["hybrid_score"] = args.alpha * dense_normalized[index] + (1.0 - args.alpha) * sparse_normalized[index]
            ranked = sorted(candidates, key=lambda doc: (-doc["hybrid_score"], -doc["dense_score"], doc["doc_id"]))
            gold = support_titles(row)
            answer = str(row.get("answer", "")).strip().lower()
            qid = query_id(row)
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            for size in (10, 20):
                selected = ranked[:size]
                payload = {"query_id": qid, "dataset": args.dataset, "pool_size": len(selected), "documents": selected}
                handles[size].write(json.dumps(payload, ensure_ascii=False) + "\n")
                selected_titles = {normalize_title(doc["title"]) for doc in selected}
                redundancy = [jaccard(f"{a['title']} {a['text']}", f"{b['title']} {b['text']}") for a, b in combinations(selected, 2)]
                statistics.append({
                    "query_id": qid,
                    "pool_size_target": size,
                    "pool_size_actual": len(selected),
                    "support_document_recall": len(gold & selected_titles) / len(gold) if gold else "",
                    "all_support_documents_present": int(bool(gold) and gold <= selected_titles),
                    "answer_document_present": int(bool(answer) and any(answer in f"{doc['title']} {doc['text']}".lower() for doc in selected)),
                    "mean_pair_redundancy": float(np.mean(redundancy)) if redundancy else 0.0,
                    "mean_sparse_score": float(np.mean([doc["sparse_score"] for doc in selected])) if selected else 0.0,
                    "mean_dense_score": float(np.mean([doc["dense_score"] for doc in selected])) if selected else 0.0,
                    "retrieval_latency_ms": elapsed_ms,
                })
    finally:
        connection.close()
        for handle in handles.values():
            handle.close()

    stats_path = args.output_dir / "retrieval_pool_statistics.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]) if statistics else ["query_id"])
        writer.writeheader()
        writer.writerows(statistics)
    manifest = {"dataset": args.dataset, "split": str(args.split.resolve()), "index": str(args.index.resolve()), "encoder": args.encoder, "alpha": args.alpha, "sparse_candidates": args.sparse_candidates, "queries": len(statistics) // 2, "outputs": {str(size): str(path.resolve()) for size, path in output_paths.items()}}
    (args.output_dir / "candidate_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

