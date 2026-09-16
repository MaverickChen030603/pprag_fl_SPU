#!/usr/bin/env python3
"""Train a RAGRoute-style source router and route the frozen R5 candidate pool.

The model consumes only query embedding, full-source centroid, and a source-ID
one-hot vector.  It never consumes the 18-float probe or any R5 labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_encoder(model_name: str, revision: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, revision=revision), AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()


def encode(tokenizer: Any, model: Any, texts: list[str], device: str, batch_size: int) -> np.ndarray:
    import torch
    outputs = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            value = model(**encoded).last_hidden_state[:, 0]
            value = torch.nn.functional.normalize(value, p=2, dim=1)
        outputs.append(value.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def support_documents(row: dict[str, Any], dataset: str, document_id: Any) -> set[str]:
    if dataset == "musique":
        return {
            document_id(dataset, str(item.get("title", "")), str(item.get("paragraph_text", "")))
            for item in row.get("paragraphs", [])
            if item.get("is_supporting", item.get("is_support", False))
        }
    facts = row.get("supporting_facts", {})
    titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
    return {document_id(dataset, str(title)) for title in titles}


def features(query_embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    count, dim = query_embeddings.shape
    clients = centroids.shape[0]
    query = np.repeat(query_embeddings, clients, axis=0)
    source = np.tile(centroids, (count, 1))
    source_id = np.tile(np.eye(clients, dtype=np.float32), (count, 1))
    return np.concatenate((query, source, source_id), axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--r5-packets", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--v16-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    sys.path.insert(0, str(args.v16_eval))
    from eval_common import document_id

    train_rows = list(rows(args.train))
    centroids = np.load(args.centroids).astype(np.float32)
    clients = centroids.shape[0]
    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment)}
    tokenizer, encoder = load_encoder(args.model, args.revision, args.device)
    train_embeddings = encode(tokenizer, encoder, [str(row["question"]) for row in train_rows], args.device, args.batch_size)
    train_x = features(train_embeddings, centroids)
    labels = []
    queries_with_support = 0
    for row in train_rows:
        support_clients = {assignment[doc] for doc in support_documents(row, args.dataset, document_id) if doc in assignment}
        queries_with_support += bool(support_clients)
        labels.extend(int(client in support_clients) for client in range(clients))
    target = np.asarray(labels, dtype=np.int64)
    if target.min() == target.max():
        raise ValueError("router training labels contain one class")
    classifier = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam", alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=args.epochs, random_state=args.seed, early_stopping=False)
    classifier.fit(train_x, target)
    train_probabilities = classifier.predict_proba(train_x)[:, 1]

    packets = list(rows(args.r5_packets))
    packet_ids = [str(row["query_id"]) for row in packets]
    r5_embeddings = encode(tokenizer, encoder, [str(row["question"]) for row in packets], args.device, args.batch_size)
    r5_x = features(r5_embeddings, centroids)
    all_scores = classifier.predict_proba(r5_x)[:, 1].reshape(len(packets), clients)
    routes = []
    for packet, scores in zip(packets, all_scores):
        candidates = [int(item["client_id"]) for item in packet["p0_candidate_records"]]
        if len(candidates) != 8 or len(set(candidates)) != 8:
            raise ValueError(f"invalid frozen candidate pool for {packet['query_id']}")
        selected = sorted(candidates, key=lambda client: (-float(scores[client]), client))[:3]
        routes.append({
            "dataset": args.dataset,
            "query_id": str(packet["query_id"]),
            "method": "b3_ragroute_protocol_adapted",
            "selected_clients": selected,
            "candidate_clients": candidates,
            "candidate_scores": {str(client): float(scores[client]) for client in candidates},
            "feature_contract": "query_bge_plus_full_source_bge_centroid_plus_source_id_onehot",
            "probe_features_used": False,
            "r5_labels_used": False,
            "reader_started": False,
        })
    args.output_dir.mkdir(parents=True)
    route_path = args.output_dir / "r5_routes_unlabeled.jsonl"
    with route_path.open("x", encoding="utf-8") as handle:
        for row in routes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    model_path = args.output_dir / "ragroute_mlp.pkl"
    with model_path.open("xb") as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "status": "trained_and_routed_unlabeled",
        "method": "b3_ragroute_protocol_adapted",
        "dataset": args.dataset,
        "external_method": "RAGRoute-style query/source-centroid/source-ID shallow MLP",
        "adaptation": "support-client supervision on this project training split; frozen R5 Top-8 restriction only at inference",
        "seed": args.seed,
        "epochs_max": args.epochs,
        "train_queries": len(train_rows),
        "train_queries_with_mapped_support_client": queries_with_support,
        "candidate_rows": int(target.size),
        "positive_rows": int(target.sum()),
        "feature_dim": int(train_x.shape[1]),
        "centroids_sha256": sha256(args.centroids),
        "assignment_sha256": sha256(args.assignment),
        "train_sha256": sha256(args.train),
        "r5_packets_sha256": sha256(args.r5_packets),
        "r5_routes_sha256": sha256(route_path),
        "model_sha256": sha256(model_path),
        "train_auc": float(roc_auc_score(target, train_probabilities)),
        "train_auprc": float(average_precision_score(target, train_probabilities)),
        "r5_queries": len(packet_ids),
        "r5_labels_opened": False,
        "reader_started": False,
        "evidence_role": "retrospective_post_hoc_r5_revealed",
        "claim_restriction": "diagnostic_only_not_fresh_confirmation",
    }
    atomic_json(args.output_dir / "training_and_route_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
