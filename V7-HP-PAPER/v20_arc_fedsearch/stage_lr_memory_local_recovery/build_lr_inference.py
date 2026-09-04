#!/usr/bin/env python3
"""Freeze LR selections, anchors, and physical-local rankings without gold metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TOKEN = re.compile(r"[A-Za-z0-9]+")


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


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
    return [dict(zip(names, values)) for values in connection.execute(sql, (query, limit))]


def compact(doc: dict[str, Any], client: int, rank: int, score: float) -> dict[str, Any]:
    return {
        "doc_id": str(doc["doc_id"]), "title": str(doc["title"]), "client_id": client,
        "local_rank": rank, "local_score": float(score), "dense_score": float(doc["dense_score"]),
        "sparse_score": float(doc["sparse_score"]),
    }


def ranks(values: list[dict[str, Any]], client: int, dense_limit: int, sparse_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dense = sorted(values, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))
    sparse = sorted(values, key=lambda doc: (-float(doc["sparse_score"]), str(doc["doc_id"])))
    return (
        [compact(doc, client, rank, float(doc["dense_score"])) for rank, doc in enumerate(dense[:dense_limit])],
        [compact(doc, client, rank, float(doc["sparse_score"])) for rank, doc in enumerate(sparse[:sparse_limit])],
    )


def sanitize(text: str, limit: int) -> str:
    values = TOKEN.findall(" ".join(text.replace("[", " ").replace("]", " ").split()))
    return " ".join(values[:limit])


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--phase", choices=("lr0", "lr1"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--b0-candidates", type=Path, required=True)
    parser.add_argument("--b0-oracle", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lr0-rankings", type=Path)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--sparse-candidates", type=int, default=100)
    parser.add_argument("--anchor-max-tokens", type=int, default=48)
    args = parser.parse_args()

    source = list(jsonl(args.split))
    if args.max_queries:
        source = source[:args.max_queries]
    questions = {qid(row): str(row["question"]) for row in source}
    candidates = {(str(row["query_id"]), row["candidate_method"]): row for row in csv_rows(args.b0_candidates)}
    oracle = {
        str(row["query_id"]): row
        for row in csv_rows(args.b0_oracle)
        if row["candidate_method"] == "REMP_rrf_p0_dense_lexical" and row["selection"] == "client_oracle_subset"
    }
    selections: list[dict[str, Any]] = []
    for query_id in questions:
        p0 = json.loads(candidates[(query_id, "P0_single_centroid")]["candidate_clients_top5"])
        remp = json.loads(candidates[(query_id, "REMP_rrf_p0_dense_lexical")]["candidate_clients_top5"])
        selections.append({"dataset": args.dataset, "query_id": query_id, "track": "D0_P0_naive_top3",
                           "selected_clients": p0[:3], "candidate_top5": p0, "gold_used_for_selection": False})
        selections.append({"dataset": args.dataset, "query_id": query_id, "track": "D1_REMP_naive_top3",
                           "selected_clients": remp[:3], "candidate_top5": remp, "gold_used_for_selection": False})
        if query_id not in oracle:
            raise KeyError(f"missing frozen R2-B0 client oracle for {query_id}")
        selections.append({"dataset": args.dataset, "query_id": query_id, "track": "O1_REMP_client_oracle_subset",
                           "selected_clients": json.loads(oracle[query_id]["selected_clients"]), "candidate_top5": remp,
                           "gold_used_for_selection": True, "diagnostic_only": True})

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "track_selections.jsonl", selections)
    unique = sorted({(row["query_id"], int(client)) for row in selections for client in row["selected_clients"]})
    profile_rows = json.loads(args.profiles.read_text(encoding="utf-8"))["profiles"]
    profiles = {int(row["client_id"]): row for row in profile_rows}

    if args.phase == "lr0":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.encoder, device=args.device)
        connections = {client: sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite") for _query, client in unique}
        emitted, timings = [], []
        try:
            for index, (query_id, client) in enumerate(unique, start=1):
                started = time.perf_counter()
                question = questions[query_id]
                values = sparse_search(connections[client], question, args.sparse_candidates)
                embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
                doc_embeddings = model.encode([f"{doc['title']}. {doc['text']}" for doc in values], normalize_embeddings=True,
                                              convert_to_numpy=True, batch_size=256, show_progress_bar=False)
                for doc, score in zip(values, (doc_embeddings @ embedding).astype(float).tolist()):
                    doc["dense_score"] = score
                q0_dense, q0_bm25 = ranks(values, client, 50, 50)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                emitted.append({"dataset": args.dataset, "query_id": query_id, "client_id": client,
                                "q0_dense_top50": q0_dense, "q0_bm25_top50": q0_bm25,
                                "physical_candidate_count": len(values), "gold_or_answer_used": False})
                timings.append({"dataset": args.dataset, "query_id": query_id, "client_id": client,
                                "phase": "lr0", "q0_elapsed_ms": elapsed_ms})
                if index % 25 == 0:
                    print(json.dumps({"phase": "lr0", "completed_query_clients": index, "target": len(unique)}), flush=True)
        finally:
            for connection in set(connections.values()):
                connection.close()
        # Ranking outputs are canonical evidence; run-specific telemetry remains separate.
        write_jsonl(args.output_root / "local_rankings_lr0.jsonl", emitted)
        write_jsonl(args.output_root / "local_timing_lr0.jsonl", timings)
        manifest = {"dataset": args.dataset, "phase": "lr0", "queries": len(questions), "unique_query_clients": len(unique),
                    "local_index_only": True, "gold_or_answer_used": False, "reader_started": False, "final_test_accessed": False}
    else:
        if args.lr0_rankings is None:
            raise ValueError("--lr0-rankings is required for lr1")
        from sentence_transformers import SentenceTransformer
        q0 = {(str(row["query_id"]), int(row["client_id"])): row for row in jsonl(args.lr0_rankings)}
        model = SentenceTransformer(args.encoder, device=args.device)
        connections = {client: sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite") for _query, client in unique}
        anchors, emitted, timings = [], [], []
        try:
            for index, (query_id, client) in enumerate(unique, start=1):
                started = time.perf_counter()
                question = questions[query_id]
                query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
                units = profiles[client]["representative_units"]
                vectors = np.asarray([unit["embedding"] for unit in units], dtype=np.float32)
                unit_index = int(np.argmax(vectors @ query_embedding))
                unit = units[unit_index]
                anchor_text = sanitize(str(unit["text"]), args.anchor_max_tokens)
                q1 = question if not anchor_text else f"{question} [ANCHOR] {anchor_text}"
                values = sparse_search(connections[client], q1, args.sparse_candidates)
                q1_embedding = model.encode([q1], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
                doc_embeddings = model.encode([f"{doc['title']}. {doc['text']}" for doc in values], normalize_embeddings=True,
                                              convert_to_numpy=True, batch_size=256, show_progress_bar=False)
                for doc, score in zip(values, (doc_embeddings @ q1_embedding).astype(float).tolist()):
                    doc["dense_score"] = score
                q1_dense, q1_bm25 = ranks(values, client, 25, 25)
                anchor = {"dataset": args.dataset, "query_id": query_id, "client_id": client, "anchor_id": unit["unit_id"],
                          "anchor_type": unit["unit_type"], "anchor_source_doc_id": unit["source_doc_id"],
                          "anchor_text_sha256": hashlib.sha256(anchor_text.encode()).hexdigest(),
                          "anchor_token_count": len(anchor_text.split()), "q1": q1, "gold_or_answer_used": False}
                anchors.append(anchor)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                emitted.append(q0[(query_id, client)] | {"q1_dense_top25": q1_dense, "q1_bm25_top25": q1_bm25})
                timings.append({"dataset": args.dataset, "query_id": query_id, "client_id": client,
                                "phase": "lr1", "q1_elapsed_ms": elapsed_ms})
                if index % 25 == 0:
                    print(json.dumps({"phase": "lr1", "completed_query_clients": index, "target": len(unique)}), flush=True)
        finally:
            for connection in set(connections.values()):
                connection.close()
        write_jsonl(args.output_root / "memory_anchors.jsonl", anchors)
        write_jsonl(args.output_root / "local_rankings_lr1.jsonl", emitted)
        write_jsonl(args.output_root / "local_timing_lr1.jsonl", timings)
        manifest = {"dataset": args.dataset, "phase": "lr1", "queries": len(questions), "unique_query_clients": len(unique),
                    "local_index_only": True, "gold_or_answer_used": False, "reader_started": False, "final_test_accessed": False}
    (args.output_root / f"{args.phase}_inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
