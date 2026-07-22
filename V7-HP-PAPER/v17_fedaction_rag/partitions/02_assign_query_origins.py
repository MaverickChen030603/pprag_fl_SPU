#!/usr/bin/env python3
"""Assign query origins from query-to-client centroid similarity without labels."""

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


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def load_partitions(root: Path) -> list[dict[str, Any]]:
    outputs = []
    for name in ("topic_silo_manifest.json", "entity_community_manifest.json", "random_control_manifest.json", "dirichlet_manifest.json"):
        path = root / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        outputs.extend(payload.get("datasets", {}).values())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).with_name("client_query_distribution.csv"))
    parser.add_argument("--output-manifest", type=Path, default=Path(__file__).with_name("query_origin_manifest.json"))
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.encoder, device=args.device)
    output_rows = []
    rng = np.random.default_rng(args.seed)
    for spec in load_partitions(args.partition_root):
        dataset, partition = spec["dataset"], spec["partition"]
        centroids = np.load(spec["centroid_path"])
        for split in ("train", "development", "calibration"):
            source = args.data_root / dataset / f"{split}.jsonl"
            source_rows = list(rows(source))
            questions = [str(row["question"]) for row in source_rows]
            embeddings = model.encode(questions, normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
            similarities = embeddings @ centroids.T
            shifted = similarities / args.temperature
            shifted -= shifted.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for row, scores, probs in zip(source_rows, similarities, probabilities):
                origin = int(rng.choice(len(probs), p=probs))
                output_rows.append({
                    "dataset": dataset,
                    "partition": partition,
                    "split": split,
                    "query_id": qid(row),
                    "origin_client": origin,
                    "origin_probability": float(probs[origin]),
                    "max_similarity": float(scores.max()),
                    "gold_labels_used": False,
                })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    counts: dict[str, dict[str, int]] = {}
    for row in output_rows:
        key = f"{row['dataset']}::{row['partition']}::{row['split']}"
        values = counts.setdefault(key, {})
        client = str(row["origin_client"])
        values[client] = values.get(client, 0) + 1
    manifest = {
        "status": "complete",
        "encoder": args.encoder,
        "temperature": args.temperature,
        "seed": args.seed,
        "gold_labels_used": False,
        "rows": len(output_rows),
        "distribution": counts,
        "output": str(args.output_csv.resolve()),
    }
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(output_rows), "output": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
