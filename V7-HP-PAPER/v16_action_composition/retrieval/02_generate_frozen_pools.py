#!/usr/bin/env python3
"""Generate strict no-injection hybrid Top-10/20 pools from a frozen index."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from retrieval_common import hop_count_without_labels, iter_rows, lexical_tokens, minmax, query_id


def fts_query(question: str) -> str:
    tokens = list(dict.fromkeys(lexical_tokens(question)))[:32]
    return " OR ".join(f'"{token}"' for token in tokens)


def search(connection: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    query = fts_query(question)
    if not query:
        return []
    sql = """
      SELECT d.doc_id, d.title, d.text, -bm25(docs_fts, 2.0, 1.0) AS sparse_score
      FROM docs_fts JOIN docs d ON d.id=docs_fts.rowid
      WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts, 2.0, 1.0) LIMIT ?
    """
    return [dict(zip(("doc_id", "title", "text", "sparse_score"), values)) for values in connection.execute(sql, (query, limit))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
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
    paths = {size: args.output_dir / f"top{size}.jsonl" for size in (10, 20)}
    handles = {size: path.open("w", encoding="utf-8") for size, path in paths.items()}
    timing, count = [], 0
    try:
        for offset, row in enumerate(iter_rows(args.split)):
            if args.max_queries is not None and offset >= args.max_queries:
                break
            started = time.perf_counter()
            question = str(row["question"])
            candidates = search(connection, question, args.sparse_candidates)
            if len(candidates) < 20:
                raise RuntimeError(f"{query_id(row)}: only {len(candidates)} corpus matches; random/gold padding is prohibited")
            query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            texts = [f"{doc['title']}. {doc['text']}" for doc in candidates]
            doc_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
            dense = (doc_embeddings @ query_embedding).astype(float).tolist()
            sparse = [float(doc["sparse_score"]) for doc in candidates]
            dense_norm, sparse_norm = minmax(dense), minmax(sparse)
            for index, doc in enumerate(candidates):
                doc["dense_score"] = dense[index]
                doc["sparse_score"] = sparse[index]
                doc["hybrid_score"] = args.alpha * dense_norm[index] + (1.0 - args.alpha) * sparse_norm[index]
            ranked = sorted(candidates, key=lambda doc: (-doc["hybrid_score"], -doc["dense_score"], doc["doc_id"]))
            elapsed = 1000.0 * (time.perf_counter() - started)
            for size in (10, 20):
                selected = ranked[:size]
                payload = {
                    "query_id": query_id(row), "dataset": args.dataset, "pool_size": size,
                    "pool": selected, "baseline_doc_ids": [doc["doc_id"] for doc in selected[:5]],
                    "hop_count": hop_count_without_labels(row, args.dataset),
                    "question_type": row.get("type", row.get("question_type", "unknown")),
                    "retrieval_latency_ms": elapsed,
                }
                handles[size].write(json.dumps(payload, ensure_ascii=False) + "\n")
            timing.append(elapsed)
            count += 1
    finally:
        connection.close()
        for handle in handles.values():
            handle.close()
    manifest = {"status": "complete", "dataset": args.dataset, "queries": count, "split": str(args.split.resolve()), "index": str(args.index.resolve()), "encoder": args.encoder, "alpha": args.alpha, "sparse_candidates": args.sparse_candidates, "gold_or_local_injection": False, "random_padding": False, "mean_retrieval_ms": sum(timing) / len(timing) if timing else None, "outputs": {str(size): str(path.resolve()) for size, path in paths.items()}}
    (args.output_dir / "candidate_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
