#!/usr/bin/env python3
"""Build label-free P0 and frozen REM-P Top-5 candidate lists for R2-B0."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
CAP = re.compile(r"(?:[A-Z][A-Za-z0-9'-]*(?:\s+|$)){1,5}")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def query_views(question: str) -> list[str]:
    entities = [value.strip() for value in CAP.findall(question) if value.strip()]
    clauses = [
        value.strip()
        for value in re.split(r"\b(?:and|or|but|than|while|that|which|who|where)\b|[;,?]", question, flags=re.I)
        if len(value.strip()) > 5
    ]
    relations = [
        value.strip()
        for value in re.findall(r"(?:of|in|by|from|with|for|between)\s+([A-Za-z][A-Za-z0-9' -]{2,40})", question, re.I)
    ]
    return [question] + entities + clauses + relations


def tokens(values: list[str]) -> set[str]:
    return {token.lower() for value in values for token in TOKEN.findall(value) if len(token) > 2}


def rank_desc(scores: list[float]) -> list[int]:
    return [int(value) for value in np.argsort(-np.asarray(scores, dtype=np.float32))]


def rrf_rank(*rankings: list[int], clients: int = 20, k: int = 60) -> list[int]:
    rank_maps = [{client: rank for rank, client in enumerate(ranking)} for ranking in rankings]
    return sorted(range(clients), key=lambda client: (-sum(1.0 / (k + rank_map[client]) for rank_map in rank_maps), client))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))["profiles"]
    source = list(rows(args.split))
    if args.max_queries:
        source = source[: args.max_queries]
    model = SentenceTransformer(args.encoder, device=args.device)
    all_terms = [set(profile["lexical_memory"]["term_counts"]) for profile in profiles]
    document_frequency = {term: sum(term in client_terms for client_terms in all_terms) for client_terms in all_terms for term in client_terms}

    emitted: list[dict[str, Any]] = []
    for row in source:
        question = str(row["question"])
        views = query_views(question)
        embeddings = model.encode(views, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        query_terms = tokens(views)
        p0_scores, dense_scores, lexical_scores = [], [], []
        for profile in profiles:
            p0_scores.append(float(embeddings[0] @ np.asarray(profile["p0_single_centroid"], dtype=np.float32)))
            unit_embeddings = np.asarray([unit["embedding"] for unit in profile["representative_units"]], dtype=np.float32)
            similarities = embeddings @ unit_embeddings.T
            dense_scores.append(float(np.max(similarities)) if similarities.size else -1.0)
            term_counts = profile["lexical_memory"]["term_counts"]
            lexical = sum(
                math.log1p(float(term_counts[term])) * math.log(21.0 / (1.0 + document_frequency.get(term, 0)))
                for term in query_terms if term in term_counts
            )
            joined = " ".join(views).lower()
            lexical += sum(0.5 * math.log1p(float(freq)) for entity, freq in profile["lexical_memory"].get("entity_frequency_sketch", {}).items() if entity.lower() in joined)
            lexical_scores.append(float(lexical))
        p0_rank = rank_desc(p0_scores)
        remp_rank = rrf_rank(p0_rank, rank_desc(dense_scores), rank_desc(lexical_scores), clients=len(profiles))
        emitted.extend(
            {
                "dataset": args.dataset,
                "query_id": qid(row),
                "candidate_method": method,
                "candidate_clients_top5": json.dumps(ranking[:5]),
                "candidate_clients_full_rank": json.dumps(ranking),
                "gold_or_answer_used_for_ranking": False,
                "reader_started": False,
            }
            for method, ranking in (("P0_single_centroid", p0_rank), ("REMP_rrf_p0_dense_lexical", remp_rank))
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "frozen_candidates.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(emitted[0]))
        writer.writeheader()
        writer.writerows(emitted)
    manifest = {
        "stage": "R2-B0",
        "dataset": args.dataset,
        "queries": len(source),
        "methods": ["P0_single_centroid", "REMP_rrf_p0_dense_lexical"],
        "candidate_L": 5,
        "gold_or_answer_used_for_ranking": False,
        "reader_started": False,
        "final_test_accessed": False,
    }
    (args.output_root / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
