#!/usr/bin/env python3
"""Freeze P0 and REM-P candidate rankings before H0 gold evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
CAP = re.compile(r"(?:[A-Z][A-Za-z0-9'-]*(?:\s+|$)){1,5}")


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def query_views(question: str) -> dict[str, Any]:
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
    return {"full_query": question, "entity_views": entities, "clause_views": clauses, "relation_views": relations}


def terms(values: list[str]) -> set[str]:
    return {value.lower() for text in values for value in TOKEN.findall(text) if len(value) > 2}


def rank(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda client: (-float(scores[client]), client))


def rrf_rank(*rankings: list[int], k: int = 60) -> list[int]:
    maps = [{client: position for position, client in enumerate(ranking)} for ranking in rankings]
    return sorted(range(20), key=lambda client: (-sum(1.0 / (k + mapping[client]) for mapping in maps), client))


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


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

    data = list(jsonl(args.split))
    if args.max_queries:
        data = data[: args.max_queries]
    profile_payload = json.loads(args.profiles.read_text(encoding="utf-8"))
    profiles = profile_payload["profiles"]
    if len(profiles) != 20:
        raise ValueError(f"expected 20 client profiles, found {len(profiles)}")
    all_terms = [set(profile["lexical_memory"]["term_counts"]) for profile in profiles]
    df = {term: sum(term in client_terms for client_terms in all_terms) for client_terms in all_terms for term in client_terms}
    model = SentenceTransformer(args.encoder, device=args.device)
    rankings_out, views_out, timing_out = [], [], []
    for position, row in enumerate(data, start=1):
        started = time.perf_counter()
        qid = query_id(row)
        view = query_views(str(row["question"]))
        strings = [view["full_query"]] + view["entity_views"] + view["clause_views"] + view["relation_views"]
        embeddings = model.encode(strings, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        query_terms = terms(strings)
        p0_scores, dense_scores, lexical_scores = [], [], []
        for profile in profiles:
            p0_scores.append(float(embeddings[0] @ np.asarray(profile["p0_single_centroid"], dtype=np.float32)))
            units = np.asarray([unit["embedding"] for unit in profile["representative_units"]], dtype=np.float32)
            similarity = embeddings @ units.T
            unit_max = similarity.max(axis=0)
            dense_scores.append(float(unit_max.max()))
            lexical = 0.0
            counts = profile["lexical_memory"]["term_counts"]
            for term in query_terms:
                if term in counts:
                    lexical += math.log1p(float(counts[term])) * math.log(21.0 / (1.0 + df.get(term, 0)))
            visible = " ".join(strings).lower()
            for entity, frequency in profile["lexical_memory"].get("entity_frequency_sketch", {}).items():
                if entity.lower() in visible:
                    lexical += 0.5 * math.log1p(float(frequency))
            lexical_scores.append(float(lexical))
        p0_ranking = rank(p0_scores)
        remp_ranking = rrf_rank(p0_ranking, rank(dense_scores), rank(lexical_scores))
        rankings_out.append({
            "dataset": args.dataset,
            "query_id": qid,
            "methods": {
                "P0_single_centroid": p0_ranking,
                "REMP_rrf_p0_dense_lexical": remp_ranking,
            },
            "gold_or_answer_used": False,
        })
        views_out.append({"dataset": args.dataset, "query_id": qid, **view, "gold_or_answer_used": False})
        timing_out.append({"dataset": args.dataset, "query_id": qid, "candidate_inference_elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
        if position % 25 == 0:
            print(json.dumps({"phase": "h0_inference", "completed_queries": position, "target": len(data)}), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "candidate_rankings.jsonl", rankings_out)
    write_jsonl(args.output_root / "query_views.jsonl", views_out)
    write_jsonl(args.output_root / "candidate_timing.jsonl", timing_out)
    manifest = {
        "stage": "CTD-CSR-H0",
        "dataset": args.dataset,
        "queries": len(data),
        "methods": ["P0_single_centroid", "REMP_rrf_p0_dense_lexical"],
        "candidate_cutoffs": [3, 5, 8],
        "profile_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
        "gold_or_answer_used": False,
        "reader_started": False,
        "final_test_accessed": False,
    }
    (args.output_root / "h0_inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
