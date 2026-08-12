#!/usr/bin/env python3
"""Replay the frozen V17 origin RNG before assigning untouched Hotpot final queries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ORDER = ("musique", "2wikimultihopqa", "hotpotqa")
SPLITS = ("train", "development", "calibration")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row):
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(20260723)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=args.device)
    existing = {}
    with (args.v17 / "partitions/client_query_distribution.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["partition"] == "topic_silo":
                existing[(row["dataset"], row["split"], row["query_id"])] = int(row["origin_client"])
    validated = 0
    for dataset in ORDER:
        centroids = np.load(args.v17 / f"partitions/centroids/{dataset}/topic_silo_m20.npy")
        for split in SPLITS:
            values = list(rows(args.v17 / f"data/{dataset}/{split}.jsonl"))
            embeddings = model.encode([str(row["question"]) for row in values], normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
            similarities = embeddings @ centroids.T
            shifted = similarities / .15
            shifted -= shifted.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
            for row, probs in zip(values, probabilities):
                choice = int(rng.choice(len(probs), p=probs))
                if existing[(dataset, split, qid(row))] != choice:
                    raise RuntimeError(f"V17 origin replay mismatch at {dataset}/{split}/{qid(row)}")
                validated += 1
    final_all = list(rows(args.v17 / "data/hotpotqa/final_test_inputs.jsonl"))
    sample_ids = {qid(row) for row in rows(args.sample)}
    centroids = np.load(args.v17 / "partitions/centroids/hotpotqa/topic_silo_m20.npy")
    embeddings = model.encode([str(row["question"]) for row in final_all], normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
    similarities = embeddings @ centroids.T
    shifted = similarities / .15; shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with args.output.open("x", encoding="utf-8") as handle:
        for row, scores, probs in zip(final_all, similarities, probabilities):
            origin = int(rng.choice(len(probs), p=probs))
            if qid(row) not in sample_ids:
                continue
            others = [int(value) for value in np.argsort(-scores) if int(value) != origin]
            payload = {"query_id": qid(row), "dataset": "hotpotqa", "partition": "topic_silo", "origin_client": origin,
                       "selected_clients": [origin, *others[:2]], "client_budget": 3,
                       "router": "exact_V17_origin_rng_replay_plus_centroid", "seed": 20260723, "temperature": .15,
                       "prior_origin_assignments_replayed": validated, "gold_or_answer_fields_used": False}
            handle.write(json.dumps(payload) + "\n"); emitted += 1
    if emitted != 300:
        raise ValueError(f"expected 300 Hotpot final origins, emitted {emitted}")
    args.output.with_suffix(".manifest.json").write_text(json.dumps({"status": "complete", "queries": emitted, "prior_assignments_validated": validated, "exact_replay": True, "labels_used": False}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "queries": emitted, "validated": validated}, indent=2))


if __name__ == "__main__":
    main()
