#!/usr/bin/env python3
"""Build label-free topic, entity-community, Dirichlet, and random silos."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’-]*(?:\s+[A-Z][A-Za-z0-9'’-]*){0,3}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_documents(index: Path) -> list[dict[str, str]]:
    connection = sqlite3.connect(index)
    try:
        rows = connection.execute("SELECT doc_id,title,text FROM docs ORDER BY doc_id").fetchall()
    finally:
        connection.close()
    return [{"doc_id": str(row[0]), "title": str(row[1]), "text": str(row[2])} for row in rows]


def encode_documents(docs: list[dict[str, str]], model_name: str, device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    texts = [f"{doc['title']}. {doc['text'][:1200]}" for doc in docs]
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)


def balanced_random(n: int, m: int, seed: int) -> np.ndarray:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    labels = np.empty(n, dtype=np.int32)
    for rank, index in enumerate(order):
        labels[index] = rank % m
    return labels


def topic_labels(embeddings: np.ndarray, m: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=m,
        random_state=seed,
        batch_size=min(4096, max(256, len(embeddings))),
        n_init=5,
        reassignment_ratio=0.01,
    )
    return model.fit_predict(embeddings).astype(np.int32)


def entities(doc: dict[str, str]) -> list[str]:
    values = [doc["title"]]
    values.extend(ENTITY_RE.findall(doc["text"][:900]))
    cleaned = []
    for value in values:
        value = " ".join(value.lower().split())
        if len(value) >= 3 and value not in cleaned:
            cleaned.append(value)
    return cleaned[:10]


def entity_community_labels(docs: list[dict[str, str]], m: int, seed: int, max_edges: int) -> tuple[np.ndarray, dict[str, Any]]:
    import networkx as nx

    doc_entities = [entities(doc) for doc in docs]
    frequency = Counter(entity for values in doc_entities for entity in values)
    allowed = {entity for entity, count in frequency.items() if 2 <= count <= 250}
    graph = nx.Graph()
    edge_count = 0
    for values in doc_entities:
        selected = [value for value in values if value in allowed][:8]
        graph.add_nodes_from(selected)
        for left, right in combinations(selected, 2):
            if graph.has_edge(left, right):
                graph[left][right]["weight"] += 1.0
            elif edge_count < max_edges:
                graph.add_edge(left, right, weight=1.0)
                edge_count += 1
    if graph.number_of_nodes() == 0:
        return balanced_random(len(docs), m, seed), {"fallback": "balanced_random_empty_graph"}
    communities = list(nx.community.louvain_communities(graph, weight="weight", seed=seed, resolution=1.0))
    entity_to_community = {entity: index for index, community in enumerate(communities) for entity in community}
    raw_assignments = []
    raw_sizes: Counter[int] = Counter()
    for index, values in enumerate(doc_entities):
        candidates = [entity_to_community[value] for value in values if value in entity_to_community]
        raw = Counter(candidates).most_common(1)[0][0] if candidates else -(index + 1)
        raw_assignments.append(raw)
        raw_sizes[raw] += 1
    loads = [0] * m
    raw_to_client: dict[int, int] = {}
    for raw, size in sorted(raw_sizes.items(), key=lambda item: (-item[1], item[0])):
        client = min(range(m), key=lambda value: (loads[value], value))
        raw_to_client[raw] = client
        loads[client] += size
    labels = np.asarray([raw_to_client[value] for value in raw_assignments], dtype=np.int32)
    return labels, {
        "entity_nodes": graph.number_of_nodes(),
        "entity_edges": graph.number_of_edges(),
        "raw_communities": len(communities),
        "fallback": None,
    }


def dirichlet_labels(topic: np.ndarray, m: int, alpha: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.empty(len(topic), dtype=np.int32)
    for cluster in sorted(set(map(int, topic))):
        indexes = np.flatnonzero(topic == cluster)
        probabilities = rng.dirichlet(np.full(m, alpha, dtype=np.float64))
        labels[indexes] = rng.choice(m, size=len(indexes), p=probabilities)
    counts = Counter(map(int, labels))
    for client in range(m):
        if counts[client]:
            continue
        donor = max(counts, key=counts.get)
        donor_index = int(np.flatnonzero(labels == donor)[0])
        labels[donor_index] = client
        counts[donor] -= 1
        counts[client] += 1
    return labels


def centroids(embeddings: np.ndarray, labels: np.ndarray, m: int) -> np.ndarray:
    output = np.zeros((m, embeddings.shape[1]), dtype=np.float32)
    for client in range(m):
        values = embeddings[labels == client]
        if len(values):
            output[client] = values.mean(axis=0)
            norm = float(np.linalg.norm(output[client]))
            if norm:
                output[client] /= norm
    return output


def write_assignments(path: Path, docs: list[dict[str, str]], labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc, client in zip(docs, labels):
            handle.write(json.dumps({"doc_id": doc["doc_id"], "client_id": int(client)}, ensure_ascii=False) + "\n")


def stats(labels: np.ndarray, m: int) -> dict[str, Any]:
    counts = [int(np.sum(labels == client)) for client in range(m)]
    mean = sum(counts) / m
    entropy = -sum((count / len(labels)) * math.log((count / len(labels)) + 1e-12) for count in counts if count)
    return {
        "documents": len(labels),
        "client_counts": counts,
        "empty_clients": sum(count == 0 for count in counts),
        "min_client_size": min(counts),
        "max_client_size": max(counts),
        "max_to_mean_ratio": max(counts) / mean if mean else None,
        "normalized_size_entropy": entropy / math.log(m) if m > 1 else 1.0,
    }


def update_aggregate(path: Path, dataset: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        content = handle.read().strip()
        current = json.loads(content) if content else {"schema_version": 1, "datasets": {}}
        current["datasets"][dataset] = payload
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(current, indent=2) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-entity-edges", type=int, default=300000)
    args = parser.parse_args()

    docs = read_documents(args.index)
    embeddings = encode_documents(docs, args.encoder, args.device, args.batch_size)
    topic = topic_labels(embeddings, args.m, args.seed)
    entity, entity_meta = entity_community_labels(docs, args.m, args.seed, args.max_entity_edges)
    partitions: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "topic_silo": (topic, {"algorithm": "MiniBatchKMeans", "gold_labels_used": False}),
        "entity_community": (entity, {"algorithm": "Louvain entity co-occurrence communities with balanced community packing", "gold_labels_used": False, **entity_meta}),
        "random_control": (balanced_random(len(docs), args.m, args.seed), {"algorithm": "size-matched balanced random", "gold_labels_used": False}),
    }
    for alpha in (0.1, 0.3, 1.0):
        key = f"dirichlet_a{str(alpha).replace('.', 'p')}"
        partitions[key] = (dirichlet_labels(topic, args.m, alpha, args.seed + int(alpha * 1000)), {"algorithm": "topic-conditioned Dirichlet stress test", "alpha": alpha, "gold_labels_used": False})

    for name, (labels, metadata) in partitions.items():
        assignment = args.output_root / "assignments" / args.dataset / f"{name}_m{args.m}.jsonl"
        centroid_path = args.output_root / "centroids" / args.dataset / f"{name}_m{args.m}.npy"
        write_assignments(assignment, docs, labels)
        centroid_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(centroid_path, centroids(embeddings, labels, args.m))
        payload = {
            "dataset": args.dataset,
            "partition": name,
            "m": args.m,
            "seed": args.seed,
            "source_index": str(args.index.resolve()),
            "encoder": args.encoder,
            "assignment_path": str(assignment.resolve()),
            "assignment_sha256": sha256(assignment),
            "centroid_path": str(centroid_path.resolve()),
            **metadata,
            **stats(labels, args.m),
        }
        manifest_name = {
            "topic_silo": "topic_silo_manifest.json",
            "entity_community": "entity_community_manifest.json",
            "random_control": "random_control_manifest.json",
        }.get(name, "dirichlet_manifest.json")
        update_aggregate(args.output_root / manifest_name, args.dataset + (f"::{name}" if name.startswith("dirichlet") else ""), payload)
    print(json.dumps({"status": "complete", "dataset": args.dataset, "documents": len(docs), "partitions": sorted(partitions)}, indent=2))


if __name__ == "__main__":
    main()
