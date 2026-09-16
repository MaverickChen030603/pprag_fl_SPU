#!/usr/bin/env python3
"""Build full-client BGE source centroids for the protocol-adapted RAGRoute B3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def encode(model: Any, tokenizer: Any, texts: list[str], device: str, max_length: int) -> np.ndarray:
    import torch
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded).last_hidden_state[:, 0]
        output = torch.nn.functional.normalize(output, p=2, dim=1)
    return output.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--clients", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    import torch
    from transformers import AutoModel, AutoTokenizer

    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment)}
    if set(assignment.values()) != set(range(args.clients)):
        raise ValueError("assignment does not contain exactly the expected client IDs")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).to(args.device).eval()
    sums: np.ndarray | None = None
    counts = np.zeros(args.clients, dtype=np.int64)
    connection = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    try:
        cursor = connection.execute("SELECT doc_id,title,text FROM docs ORDER BY doc_id")
        batch: list[tuple[str, str]] = []
        def consume(values: list[tuple[str, str]]) -> None:
            nonlocal sums
            if not values:
                return
            embeddings = encode(model, tokenizer, [text for _, text in values], args.device, args.max_length)
            if sums is None:
                sums = np.zeros((args.clients, embeddings.shape[1]), dtype=np.float64)
            for (doc_id, _), embedding in zip(values, embeddings):
                try:
                    client = assignment[doc_id]
                except KeyError as error:
                    raise KeyError(f"index document missing from assignment: {doc_id}") from error
                sums[client] += embedding
                counts[client] += 1
        for doc_id, title, text in cursor:
            batch.append((str(doc_id), f"{title}. {text}"))
            if len(batch) == args.batch_size:
                consume(batch)
                batch = []
        consume(batch)
    finally:
        connection.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if sums is None or (counts == 0).any():
        raise ValueError("at least one client has no documents")
    centroids = sums / counts[:, None]
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "source_centroids.npy"
    with output.open("xb") as handle:
        np.save(handle, centroids.astype(np.float32))
    manifest = {
        "status": "complete_full_client_source_centroids",
        "dataset": args.dataset,
        "encoder": args.model,
        "encoder_revision": args.revision,
        "pooling": "normalized_cls_embedding_then_client_mean_then_l2_normalize",
        "documents": int(counts.sum()),
        "documents_per_client": [int(value) for value in counts],
        "clients": args.clients,
        "embedding_dim": int(centroids.shape[1]),
        "index_sha256": sha256(args.index),
        "assignment_sha256": sha256(args.assignment),
        "centroids_sha256": sha256(output),
        "r5_labels_opened": False,
        "reader_started": False,
    }
    atomic_json(args.output_dir / "centroid_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
