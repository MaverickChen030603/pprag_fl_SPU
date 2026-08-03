#!/usr/bin/env python3
"""Materialize inherited V17 origin-plus-centroid Bc=3 routes without labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def load_origins(path: Path, dataset: str) -> dict[str, int]:
    output: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["partition"] == "topic_silo" and row["split"] == "development":
                output[str(row["query_id"])] = int(row["origin_client"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--origins", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client-budget", type=int, default=3)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen routes: {args.output}")
    if args.client_budget != 3:
        raise ValueError("V20 M0-Confirm freezes Bc=3")
    source = list(rows(args.split))
    origins = load_origins(args.origins, args.dataset)
    missing = [query_id(row) for row in source if query_id(row) not in origins]
    if missing:
        raise KeyError(f"missing inherited origins for {len(missing)} development queries")
    from sentence_transformers import SentenceTransformer

    centroids = np.load(args.centroids)
    model = SentenceTransformer(args.encoder, device=args.device)
    embeddings = model.encode(
        [str(row["question"]) for row in source],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=args.batch_size,
        show_progress_bar=False,
    )
    scores = embeddings @ centroids.T
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, values in zip(source, scores):
            qid = query_id(row)
            origin = origins[qid]
            others = [int(value) for value in np.argsort(-values) if int(value) != origin]
            selected = [origin] + others[:2]
            if len(selected) != 3 or len(set(selected)) != 3:
                raise AssertionError(f"{qid}: invalid Bc=3 route")
            payload = {
                "query_id": qid,
                "dataset": args.dataset,
                "partition": "topic_silo",
                "origin_client": origin,
                "selected_clients": selected,
                "client_scores": {str(client): float(values[client]) for client in selected},
                "client_budget": 3,
                "router": "origin_plus_centroid_inherited_v17",
                "routing_features": "question_embedding_and_frozen_topic_centroids_only",
                "gold_or_answer_fields_used": False,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    manifest = {
        "status": "complete",
        "dataset": args.dataset,
        "queries": len(source),
        "router": "V17 inherited origin_plus_centroid",
        "client_budget": 3,
        "partition": "topic_silo",
        "origins": str(args.origins.resolve()),
        "centroids": str(args.centroids.resolve()),
        "gold_or_answer_fields_used": False,
        "reader_started": False,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
