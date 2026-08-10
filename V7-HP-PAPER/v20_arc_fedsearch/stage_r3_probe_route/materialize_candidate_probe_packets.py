#!/usr/bin/env python3
"""Materialize label-free R3 probe packets for frozen P0 candidate clients."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_probe_audit import (
    DEPTH,
    CLIENTS,
    entities,
    entropy,
    entity_diversity,
    query_terms,
    rank_correlation,
    sparse_search,
    title_diversity,
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def load_routes(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    return {str(item["query_id"]): [int(value) for value in item["selected_clients"]] for item in rows(path)}


def doc_entry(doc: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "doc_id": str(doc["doc_id"]),
        "local_rank": rank,
        "dense_score": float(doc["dense_score"]),
        "bm25_score": float(doc["sparse_score"]),
        "payload_bytes": len((str(doc["title"]) + "\n" + str(doc["text"])).encode("utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inherited-routes", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sparse-candidates", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="smoke-only query limit; never use for formal packets")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    data = list(rows(args.split))
    if args.limit is not None:
        data = data[:args.limit]
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))["profiles"]
    p0 = np.asarray([profile["p0_single_centroid"] for profile in profiles], dtype=np.float32)
    inherited_routes = load_routes(args.inherited_routes)
    completed = {str(item["query_id"]) for item in rows(args.output)} if args.resume and args.output.exists() else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=args.device)
    connections = {client: sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite") for client in range(CLIENTS)}
    started, emitted = time.perf_counter(), 0
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for row in data:
                query_id = qid(row)
                if query_id in completed:
                    continue
                query_started = time.perf_counter()
                question = str(row["question"])
                query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
                static_scores = (p0 @ query_embedding).astype(float).tolist()
                static_rank = [int(index) for index in np.argsort(-np.asarray(static_scores), kind="stable")]
                p0_candidates = static_rank[:8]
                materialized_clients = list(dict.fromkeys(p0_candidates + inherited_routes.get(query_id, [])))
                by_client: dict[int, list[dict[str, Any]]] = {}
                flat: list[dict[str, Any]] = []
                for client in materialized_clients:
                    values = sparse_search(connections[client], question, args.sparse_candidates)
                    for document in values:
                        document["client_id"] = client
                    by_client[client] = values
                    flat.extend(values)
                if flat:
                    embeddings = model.encode(
                        [f"{document['title']}. {document['text']}" for document in flat],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        batch_size=256,
                        show_progress_bar=False,
                    )
                    for document, score in zip(flat, (embeddings @ query_embedding).astype(float).tolist()):
                        document["dense_score"] = score
                q_terms, q_entities = query_terms(question), entities(question)
                dense_order: dict[int, list[dict[str, Any]]] = {}
                sparse_order: dict[int, list[dict[str, Any]]] = {}
                title_batch: list[str] = []
                title_clients: list[int] = []
                for client in materialized_clients:
                    dense_order[client] = sorted(by_client[client], key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))
                    sparse_order[client] = sorted(by_client[client], key=lambda doc: (-float(doc["sparse_score"]), str(doc["doc_id"])))
                    if dense_order[client]:
                        title_clients.append(client)
                        title_batch.append(str(dense_order[client][0]["title"]))
                title_similarities = {}
                if title_batch:
                    title_embeddings = model.encode(title_batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                    title_similarities = {client: float(query_embedding @ title_embedding) for client, title_embedding in zip(title_clients, title_embeddings)}
                records = []
                for client in p0_candidates:
                    dense_all, sparse_all = dense_order[client], sparse_order[client]
                    dense, sparse = dense_all[:DEPTH], sparse_all[:DEPTH]
                    dense_scores = [float(document["dense_score"]) for document in dense]
                    sparse_scores = [float(document["sparse_score"]) for document in sparse]
                    dense_ids = {str(document["doc_id"]) for document in dense[:3]}
                    sparse_ids = {str(document["doc_id"]) for document in sparse[:3]}
                    dense_rank = {str(document["doc_id"]): rank for rank, document in enumerate(dense_all)}
                    bm25_anchor = 0.0
                    if sparse_all:
                        bm25_anchor = 1.0 - dense_rank[str(sparse_all[0]["doc_id"])] / max(1, len(dense_all) - 1)
                    titles = [str(document["title"]) for document in dense[:3]]
                    records.append({
                        "client_id": client,
                        "static_score": static_scores[client],
                        "static_candidate_rank": static_rank.index(client) + 1,
                        "dense_top1_score": dense_scores[0] if dense_scores else 0.0,
                        "dense_top3_mean": float(np.mean(dense_scores[:3])) if dense_scores else 0.0,
                        "dense_top1_top2_margin": dense_scores[0] - dense_scores[1] if len(dense_scores) > 1 else 0.0,
                        "dense_score_std": float(np.std(dense_scores)) if dense_scores else 0.0,
                        "dense_score_entropy": entropy(dense_scores),
                        "dense_local_rank_percentile": bm25_anchor,
                        "bm25_top1_score": sparse_scores[0] if sparse_scores else 0.0,
                        "bm25_top3_mean": float(np.mean(sparse_scores[:3])) if sparse_scores else 0.0,
                        "bm25_top1_top2_margin": sparse_scores[0] - sparse_scores[1] if len(sparse_scores) > 1 else 0.0,
                        "dense_bm25_top1_same": int(bool(dense and sparse and dense[0]["doc_id"] == sparse[0]["doc_id"])),
                        "dense_bm25_top3_overlap": len(dense_ids & sparse_ids) / 3.0,
                        "dense_sparse_rank_correlation": rank_correlation(dense, sparse),
                        "matched_query_entity_count": len(q_entities & entities(str(dense[0]["title"]))) if dense else 0,
                        "matched_query_token_count": len(q_terms & query_terms(" ".join(titles))),
                        "matched_title_token_count": len(q_terms & query_terms(str(dense[0]["title"]))) if dense else 0,
                        "query_title_embedding_similarity": title_similarities.get(client, 0.0),
                        "top3_title_diversity": title_diversity(titles),
                        "top3_entity_diversity": entity_diversity(titles),
                        "dense_docs_top10": [doc_entry(document, rank) for rank, document in enumerate(dense)],
                    })
                deep_docs = {
                    str(client): [doc_entry(document, rank) for rank, document in enumerate(dense_order[client][:DEPTH])]
                    for client in materialized_clients
                }
                elapsed_ms = (time.perf_counter() - query_started) * 1000.0
                payload = {
                    "dataset": args.dataset,
                    "query_id": query_id,
                    "p0_candidate_clients": p0_candidates,
                    "p0_candidate_records": records,
                    "inherited_selected_clients": inherited_routes.get(query_id),
                    "local_dense_docs_top10": deep_docs,
                    "materialized_client_count": len(materialized_clients),
                    "probe_materialization_latency_ms": elapsed_ms,
                    "wire_payload_contains_text": False,
                    "wire_payload_contains_embedding": False,
                    "gold_or_answer_used": False,
                    "reader_started": False,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                emitted += 1
                if emitted % 10 == 0:
                    print(json.dumps({"dataset": args.dataset, "completed": emitted + len(completed), "target": len(data), "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    finally:
        for connection in connections.values():
            connection.close()
    manifest = {
        "dataset": args.dataset,
        "queries": emitted + len(completed),
        "candidate_L": 8,
        "local_depth": DEPTH,
        "sparse_candidates": args.sparse_candidates,
        "inherited_route_docs_materialized": bool(args.inherited_routes),
        "gold_or_answer_used": False,
        "reader_started": False,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
