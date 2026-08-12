#!/usr/bin/env python3
"""Build frozen inherited H0 routes for R5-C1 without labels or outcomes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ORDER = ("musique", "2wikimultihopqa", "hotpotqa")
SPLITS = ("train", "development", "calibration")
SEED = 20260723
TEMPERATURE = 0.15
EXPECTED = 4200
BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row):
    return str(row.get("query_id", row.get("_id", row.get("id")))).strip().lower()


def probabilities(similarities: np.ndarray) -> np.ndarray:
    shifted = similarities / TEMPERATURE
    shifted -= shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = list(rows(args.split))
    if len(selected) != EXPECTED or any(set(row) != {"query_id", "question"} for row in selected):
        raise ValueError("R5-C1 split must contain exactly 4,200 query_id/question-only rows")

    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(SEED)
    model = SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        revision=BGE_REVISION,
        device=args.device,
        local_files_only=True,
    )
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
            embeddings = model.encode(
                [str(row["question"]) for row in values],
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=256,
                show_progress_bar=False,
            )
            sims = embeddings @ centroids.T
            for row, probs in zip(values, probabilities(sims)):
                choice = int(rng.choice(len(probs), p=probs))
                if existing[(dataset, split, qid(row))] != choice:
                    raise RuntimeError(f"V17 origin replay mismatch at {dataset}/{split}/{qid(row)}")
                validated += 1

    # Consume the already-frozen V17 final-input segment before extending the
    # same RNG stream to C1 in its preregistered hash order.
    final_values = list(rows(args.v17 / "data/hotpotqa/final_test_inputs.jsonl"))
    centroids = np.load(args.v17 / "partitions/centroids/hotpotqa/topic_silo_m20.npy")
    final_embeddings = model.encode(
        [str(row["question"]) for row in final_values],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=256,
        show_progress_bar=False,
    )
    final_sims = final_embeddings @ centroids.T
    for probs in probabilities(final_sims):
        rng.choice(len(probs), p=probs)

    embeddings = model.encode(
        [str(row["question"]) for row in selected],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=256,
        show_progress_bar=False,
    )
    sims = embeddings @ centroids.T
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        for row, scores, probs in zip(selected, sims, probabilities(sims)):
            origin = int(rng.choice(len(probs), p=probs))
            others = [int(value) for value in np.argsort(-scores, kind="stable") if int(value) != origin]
            payload = {
                "query_id": qid(row),
                "dataset": "hotpotqa",
                "partition": "topic_silo",
                "origin_client": origin,
                "selected_clients": [origin, *others[:2]],
                "client_budget": 3,
                "router": "exact_V17_origin_rng_replay_then_C1_hash_order_extension",
                "seed": SEED,
                "temperature": TEMPERATURE,
                "prior_origin_assignments_replayed": validated,
                "v17_final_segment_consumed": len(final_values),
                "gold_or_answer_fields_used": False,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"status": "complete", "queries": len(selected), "validated": validated}))


if __name__ == "__main__":
    import os

    main()
