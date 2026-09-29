#!/usr/bin/env python3
"""Materialize B3F's frozen RAGRoute routes without opening fresh labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SEEDS = (0, 1, 2)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_encoder(device: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    model = "BAAI/bge-base-en-v1.5"
    revision = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, local_files_only=True
    )
    encoder = AutoModel.from_pretrained(
        model, revision=revision, local_files_only=True
    ).to(device).eval()
    return tokenizer, encoder


def encode(tokenizer: Any, model: Any, questions: list[str], device: str) -> np.ndarray:
    import torch

    values = []
    for start in range(0, len(questions), 64):
        batch = tokenizer(
            questions[start : start + 64],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            embedding = torch.nn.functional.normalize(
                model(**batch).last_hidden_state[:, 0], p=2, dim=1
            )
        values.append(embedding.float().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def load_ensemble(model_dir: Path, device: str) -> tuple[list[Any], np.ndarray, np.ndarray]:
    import torch

    sys.path.insert(0, str(model_dir.parent.parent))
    from train_b3f_ragroute import B3FRouter

    scaler = np.load(model_dir / "scaler.npz")
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    models = []
    for seed in SEEDS:
        checkpoint = torch.load(
            model_dir / f"seed_{seed}.pt", map_location=device, weights_only=True
        )
        model = B3FRouter(int(checkpoint["input_dim"])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)
    return models, mean, scale


def ensemble_scores(
    models: list[Any], features: np.ndarray, mean: np.ndarray, scale: np.ndarray, device: str
) -> np.ndarray:
    import torch

    normalized = (features - mean) / np.where(scale == 0, 1.0, scale)
    tensor = torch.from_numpy(normalized.astype(np.float32)).to(device)
    with torch.inference_mode():
        scores = [torch.sigmoid(model(tensor)).cpu().numpy() for model in models]
    return np.mean(np.stack(scores, axis=0), axis=0)


def merge(packet: dict[str, Any], clients: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [
        dict(document)
        for client in clients
        for document in packet["local_dense_docs_top10"][str(client)][:5]
    ]
    if len(documents) != 5 * len(clients):
        raise ValueError(f"{packet['query_id']}: incomplete local document payload")
    top10 = sorted(
        documents, key=lambda row: (-float(row["dense_score"]), str(row["doc_id"]))
    )[:10]
    return documents, top10


def lookup(connection: sqlite3.Connection, ids: list[str]) -> list[dict[str, str]]:
    documents = []
    for doc_id in ids:
        row = connection.execute(
            "SELECT doc_id,title,text FROM docs WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"missing canonical document {doc_id}")
        documents.append({"doc_id": str(row[0]), "title": str(row[1]), "text": str(row[2])})
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--retrieval-root",
        type=Path,
        required=True,
        help="Frozen experiment root containing the historical RAGRoute centroids.",
    )
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.stage / "contexts/b3f_ragroute_contexts_unscored.jsonl"
    manifest_path = args.stage / "contexts/b3f_ragroute_contexts_manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("B3F context output already exists; refusing to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, encoder = load_encoder(args.device)
    summary: dict[str, Any] = {}
    total = 0
    with output.open("x", encoding="utf-8") as handle:
        for dataset in DATASETS:
            blind = list(rows(args.stage / f"inputs/{dataset}_blind_n500.jsonl"))
            packets = {
                str(row["query_id"]): row
                for row in rows(args.stage / f"retrieval/{dataset}_probe_packets.jsonl")
            }
            ids = [str(row["query_id"]) for row in blind]
            if len(blind) != 500 or len(set(ids)) != 500 or set(ids) != set(packets):
                raise ValueError(f"{dataset}: frozen blind inputs and packets disagree")
            embeddings = encode(tokenizer, encoder, [str(row["question"]) for row in blind], args.device)
            centroids = np.load(
                args.retrieval_root
                / f"runs/ragroute_b3_r5_posthoc_20260916/centroids/{dataset}/source_centroids.npy"
            ).astype(np.float32)
            clients = int(centroids.shape[0])
            if clients != 8:
                raise ValueError(f"{dataset}: expected eight source centroids, found {clients}")
            models, mean, scale = load_ensemble(args.stage / f"models/{dataset}", args.device)
            selected_counts = {"ragroute_fixed3": [], "ragroute_original_threshold": []}
            index = sqlite3.connect(
                f"file:{args.base / f'inputs/indexes/{dataset}.sqlite'}?mode=ro", uri=True
            )
            try:
                for position, source in enumerate(blind):
                    query_id = str(source["query_id"])
                    question = str(source["question"])
                    packet = packets[query_id]
                    records = list(packet["p0_candidate_records"])
                    candidates = [int(row["client_id"]) for row in records]
                    if len(candidates) != 8 or len(set(candidates)) != 8:
                        raise ValueError(f"{dataset}/{query_id}: invalid frozen Top-8")
                    # Retain the frozen packet order: it is the OriginalThreshold tie/order rule.
                    features = np.concatenate(
                        (
                            np.repeat(embeddings[position : position + 1], 8, axis=0),
                            centroids[candidates],
                            np.eye(clients, dtype=np.float32)[candidates],
                        ),
                        axis=1,
                    )
                    probabilities = ensemble_scores(models, features, mean, scale, args.device)
                    scored = {client: float(probabilities[idx]) for idx, client in enumerate(candidates)}
                    methods = {
                        "ragroute_fixed3": sorted(candidates, key=lambda client: (-scored[client], client))[:3],
                        "ragroute_original_threshold": [
                            client for client in candidates if scored[client] > 0.5
                        ],
                    }
                    for method, selected in methods.items():
                        transmitted, top10 = merge(packet, selected)
                        documents = lookup(index, [str(row["doc_id"]) for row in top10[:5]])
                        context_hash = digest({"question": question, "docs": documents})
                        payload = {
                            "dataset": dataset,
                            "query_id": query_id,
                            "question": question,
                            "method": method,
                            "candidate_client_ids": candidates,
                            "selected_client_ids": selected,
                            "candidate_probabilities": {str(client): scored[client] for client in candidates},
                            "local_doc_ids": {str(client): [str(doc["doc_id"]) for doc in packet["local_dense_docs_top10"][str(client)][:10]] for client in selected},
                            "transmitted_doc_ids": [str(doc["doc_id"]) for doc in transmitted],
                            "merged_top10_doc_ids": [str(doc["doc_id"]) for doc in top10],
                            "reader_top5_doc_ids": [str(doc["doc_id"]) for doc in top10[:5]],
                            "reader_context_docs": documents,
                            "probe_packet_hash": digest(packet),
                            "context_hash": context_hash,
                            "retrieval_output_hash": digest({"method": method, "clients": selected, "top10": [str(doc["doc_id"]) for doc in top10]}),
                            "candidate_clients": 8,
                            "client_budget": len(selected),
                            "local_depth": 10,
                            "documents_per_client": 5,
                            "transmitted_documents": len(transmitted),
                            "ensemble_seeds": list(SEEDS),
                            "threshold": 0.5 if method == "ragroute_original_threshold" else None,
                            "gold_or_answer_used": False,
                            "labels_loaded": False,
                            "metrics_computed": False,
                            "reader_started": False,
                        }
                        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        selected_counts[method].append(len(selected))
                        total += 1
            finally:
                index.close()
            summary[dataset] = {
                "queries": len(blind),
                "rows": len(blind) * 2,
                "selected_client_count": {
                    method: {
                        "mean": float(np.mean(counts)),
                        "min": int(np.min(counts)),
                        "max": int(np.max(counts)),
                    }
                    for method, counts in selected_counts.items()
                },
            }
    if total != 3000:
        raise ValueError(f"expected 3,000 contexts, found {total}")
    manifest = {
        "status": "complete_blind_ragroute_context_materialization",
        "labels_loaded": False,
        "metrics_computed": False,
        "output": str(output),
        "rows": total,
        "output_sha256": sha256(output),
        "datasets": summary,
        "methods": ["ragroute_fixed3", "ragroute_original_threshold"],
        "fixed3_policy": "mean seed-0/1/2 sigmoid, descending score then client id, top 3",
        "original_threshold_policy": "mean seed-0/1/2 sigmoid, retain frozen P0 order where score > 0.5, no fallback",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
