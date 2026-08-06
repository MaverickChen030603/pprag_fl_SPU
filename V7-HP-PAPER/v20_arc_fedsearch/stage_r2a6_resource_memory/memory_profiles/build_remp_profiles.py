#!/usr/bin/env python3
"""Build fixed-size, train-corpus-only Representative Evidence Memory Profiles.

No query, support, answer, reader, calibration, or final-test field is read by
this program.  The only dataset records consumed are Router-Train contexts,
whose complete document lists define the client training corpus.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import resource
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
CAP = re.compile(r"(?:[A-Z][A-Za-z0-9'-]*(?:\s+|$)){1,6}")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
RELATION_WORDS = {"is", "was", "were", "became", "born", "died", "directed", "wrote", "written", "founded", "located", "played", "produced", "married", "served", "led"}
NEAR_DUP_JACCARD = 0.85


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def did(dataset: str, title: str, text: str = "") -> str:
    key = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(key.encode()).hexdigest()[:20]}"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def tokenize(text: str) -> list[str]:
    return [piece.lower() for piece in TOKEN.findall(text) if len(piece) > 2]


def truncate_tokens(text: str, limit: int) -> str:
    parts = TOKEN.findall(text)
    return " ".join(parts[:limit])


def title_entities(title: str) -> list[str]:
    result: list[str] = []
    for found in CAP.findall(title):
        value = " ".join(found.split()).strip()
        if len(value) >= 3 and not value.isdigit():
            result.append(value)
    return result


def train_document_ids(dataset: str, train_path: Path) -> set[str]:
    ids: set[str] = set()
    for record in rows(train_path):
        if dataset == "musique":
            for paragraph in record.get("paragraphs", []):
                ids.add(did(dataset, paragraph.get("title", ""), paragraph.get("paragraph_text", "")))
        else:
            for context in record.get("context", []):
                if context:
                    ids.add(did(dataset, str(context[0])))
    return ids


def read_client_docs(index_path: Path, allowed: set[str]) -> list[Document]:
    connection = sqlite3.connect(index_path)
    try:
        result = []
        for doc_id, title, text in connection.execute("select doc_id,title,text from docs"):
            if str(doc_id) in allowed:
                result.append(Document(str(doc_id), str(title), str(text)))
        return result
    finally:
        connection.close()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    scale = np.linalg.norm(embeddings, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return embeddings / scale


def title_unit(doc: Document) -> tuple[str, str]:
    return "title", truncate_tokens(doc.title, 16)


def entity_unit(doc: Document) -> tuple[str, str]:
    entities = title_entities(doc.title)
    return "entity_group", truncate_tokens(" ; ".join(entities) or doc.title, 12)


def relation_sentence(doc: Document) -> str:
    for sentence in SENTENCE.split(" ".join(doc.text.split())):
        terms = set(tokenize(sentence))
        if terms.intersection(RELATION_WORDS):
            return sentence
    return " ".join(doc.text.split())


def snippet_unit(doc: Document) -> tuple[str, str]:
    return "snippet", truncate_tokens(f"{doc.title}. {relation_sentence(doc)}", 32)


def relation_unit(doc: Document) -> tuple[str, str]:
    sentence = relation_sentence(doc)
    words = TOKEN.findall(sentence)
    pivot = next((i for i, word in enumerate(words) if word.lower() in RELATION_WORDS), 0)
    left = max(0, pivot - 4)
    return "rare_relation_phrase", " ".join(words[left : left + 12])


def near_duplicate(text: str, selected: list[str]) -> bool:
    normalized = norm(text)
    terms = set(tokenize(text))
    for existing in selected:
        if normalized == norm(existing):
            return True
        other = set(tokenize(existing))
        if terms and other and len(terms & other) / len(terms | other) >= NEAR_DUP_JACCARD:
            return True
    return False


def make_record(variant: str, client_id: int, ordinal: int, doc: Document, unit_type: str, unit_text: str, method: str, embedding: np.ndarray, score: float) -> dict[str, Any]:
    return {
        "unit_id": f"{variant}:c{client_id:02d}:u{ordinal:02d}",
        "variant": variant,
        "client_id": client_id,
        "unit_type": unit_type,
        "source_document_id": doc.doc_id,
        "unit_text": unit_text,
        "selection_method": method,
        "selection_score": round(float(score), 8),
        "embedding_hash": sha_array(embedding),
        "token_count": len(TOKEN.findall(unit_text)),
        "utf8_bytes": len(unit_text.encode("utf-8")),
    }


def select_medoid_indices(embeddings: np.ndarray, count: int, seed: int) -> list[int]:
    from sklearn.cluster import MiniBatchKMeans

    clusters = min(count, len(embeddings))
    model = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=3, batch_size=min(2048, max(256, len(embeddings))) )
    labels = model.fit_predict(embeddings)
    centers = normalize_embeddings(model.cluster_centers_.astype(np.float32))
    selected: list[int] = []
    for cluster in range(clusters):
        indices = np.where(labels == cluster)[0]
        local = embeddings[indices] @ centers[cluster]
        selected.append(int(indices[np.argmax(local)]))
    return selected


def select_farthest_first(embeddings: np.ndarray, count: int) -> list[int]:
    if len(embeddings) == 0:
        return []
    # Mean-vector similarity gives a central deterministic first unit without
    # materializing an O(n^2) client-by-client similarity matrix.
    centrality = embeddings @ embeddings.mean(axis=0)
    selected = [int(np.argmax(centrality))]
    min_distance = 1.0 - embeddings @ embeddings[selected[0]]
    while len(selected) < min(count, len(embeddings)):
        min_distance[selected] = -1.0
        nxt = int(np.argmax(min_distance))
        selected.append(nxt)
        min_distance = np.minimum(min_distance, 1.0 - embeddings @ embeddings[nxt])
    return selected


def idf_score(tokens: Iterable[str], client_df: dict[str, int], clients: int = 20) -> float:
    values = [math.log((clients + 1.0) / (1.0 + client_df.get(token, 0))) for token in tokens]
    return float(sum(values) / max(1, len(values)))


def discriminative_candidates(docs: list[Document], token_client_df: dict[str, int], entity_client_df: dict[str, int]) -> dict[str, list[tuple[float, int, str, str]]]:
    buckets: dict[str, list[tuple[float, int, str, str]]] = {"title": [], "entity_group": [], "snippet": [], "rare_relation_phrase": []}
    local_terms = collections.Counter(term for doc in docs for term in tokenize(doc.title + " " + doc.text))
    local_entities = collections.Counter(entity.lower() for doc in docs for entity in title_entities(doc.title))
    def representative_bonus(values: Iterable[str], table: collections.Counter[str]) -> float:
        items = list(values)
        return float(np.mean([math.log1p(table[value]) for value in items])) if items else 0.0
    for index, doc in enumerate(docs):
        unit_type, value = title_unit(doc)
        if len(tokenize(value)) >= 1:
            title_tokens = tokenize(value)
            buckets[unit_type].append((idf_score(title_tokens, token_client_df) + 0.08 * representative_bonus(title_tokens, local_terms), index, unit_type, value))
        unit_type, value = entity_unit(doc)
        entity_values = [entity.lower() for entity in title_entities(doc.title)]
        entity_score = idf_score(entity_values, entity_client_df) + 0.08 * representative_bonus(entity_values, local_entities)
        if title_entities(doc.title):
            buckets[unit_type].append((entity_score, index, unit_type, value))
        unit_type, value = snippet_unit(doc)
        relation_bonus = 0.2 if set(tokenize(value)).intersection(RELATION_WORDS) else 0.0
        snippet_tokens = tokenize(value)
        buckets[unit_type].append((idf_score(snippet_tokens, token_client_df) + 0.08 * representative_bonus(snippet_tokens, local_terms) + relation_bonus, index, unit_type, value))
        unit_type, value = relation_unit(doc)
        if set(tokenize(value)).intersection(RELATION_WORDS):
            relation_tokens = tokenize(value)
            buckets[unit_type].append((idf_score(relation_tokens, token_client_df) + 0.08 * representative_bonus(relation_tokens, local_terms) + 0.4, index, unit_type, value))
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (-row[0], docs[row[1]].doc_id, row[3]))
    return buckets


def take_diverse_records(variant: str, client_id: int, docs: list[Document], embeddings: np.ndarray, indices: list[int], method: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    selected_text: list[str] = []
    for index in indices:
        # Preserve the medoid/farthest-first *document* choice.  If two
        # selected documents have near-identical titles, expose a bounded unit
        # from the same chosen document instead of silently shrinking capacity.
        candidates = (title_unit(docs[index]), entity_unit(docs[index]), snippet_unit(docs[index]), relation_unit(docs[index]))
        for unit_type, text in candidates:
            if not text or near_duplicate(text, selected_text):
                continue
            selected_text.append(text)
            records.append(make_record(variant, client_id, len(records), docs[index], unit_type, text, method, embeddings[index], float(embeddings[index].mean())))
            vectors.append(embeddings[index])
            break
    return records, np.asarray(vectors, dtype=np.float32)


def take_discriminative_records(variant: str, client_id: int, docs: list[Document], embeddings: np.ndarray, buckets: dict[str, list[tuple[float, int, str, str]]], count: int, method: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    quotas = {"title": 10, "entity_group": 8, "snippet": 8, "rare_relation_phrase": 6}
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    used_text: list[str] = []
    used_doc_type: set[tuple[str, str]] = set()
    for kind, quota in quotas.items():
        for score, index, unit_type, text in buckets[kind]:
            key = (docs[index].doc_id, unit_type)
            if key in used_doc_type or not text or near_duplicate(text, used_text):
                continue
            used_doc_type.add(key)
            used_text.append(text)
            records.append(make_record(variant, client_id, len(records), docs[index], unit_type, text, method, embeddings[index], score))
            vectors.append(embeddings[index])
            if sum(1 for record in records if record["unit_type"] == kind) >= quota or len(records) >= count:
                break
        if len(records) >= count:
            break
    if len(records) < count:
        flat = sorted((candidate for group in buckets.values() for candidate in group), key=lambda row: (-row[0], docs[row[1]].doc_id, row[2]))
        for score, index, unit_type, text in flat:
            key = (docs[index].doc_id, unit_type)
            if key in used_doc_type or not text or near_duplicate(text, used_text):
                continue
            used_doc_type.add(key)
            used_text.append(text)
            records.append(make_record(variant, client_id, len(records), docs[index], unit_type, text, method + "_fill", embeddings[index], score))
            vectors.append(embeddings[index])
            if len(records) >= count:
                break
    return records[:count], np.asarray(vectors[:count], dtype=np.float32)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--router-train", type=Path, required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--units-per-client", type=int, default=32)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.units_per_client != 32:
        raise ValueError("R2-A.6 preregistration fixes capacity at 32 units per client")

    from sentence_transformers import SentenceTransformer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed = train_document_ids(args.dataset, args.router_train)
    train_manifest = {
        "dataset": args.dataset,
        "router_train_path": str(args.router_train.resolve()),
        "router_train_context_only": True,
        "answer_support_evidence_fields_read": False,
        "document_count": len(allowed),
        "document_ids": sorted(allowed),
        "document_ids_sha256": sha_text("\n".join(sorted(allowed))),
    }
    (args.output_dir / "train_corpus_manifest.json").write_text(json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    corpus: dict[int, list[Document]] = {}
    client_token_sets: dict[int, set[str]] = {}
    client_entity_sets: dict[int, set[str]] = {}
    for client_id in range(20):
        docs = read_client_docs(args.local_index_root / f"client_{client_id:02d}.sqlite", allowed)
        if not docs:
            raise ValueError(f"client {client_id} has no Router-Train corpus documents")
        corpus[client_id] = docs
        client_token_sets[client_id] = {term for doc in docs for term in set(tokenize(doc.title + " " + doc.text))}
        client_entity_sets[client_id] = {entity.lower() for doc in docs for entity in title_entities(doc.title)}
    token_client_df = collections.Counter(term for values in client_token_sets.values() for term in values)
    entity_client_df = collections.Counter(entity for values in client_entity_sets.values() for entity in values)

    model = SentenceTransformer(args.encoder, device=args.device)
    all_units: list[dict[str, Any]] = []
    embeddings_payload: dict[str, np.ndarray] = {}
    stats: list[dict[str, Any]] = []
    for client_id in range(20):
        started = time.perf_counter()
        docs = corpus[client_id]
        texts = [truncate_tokens(f"{doc.title}. {doc.text}", 64) for doc in docs]
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=args.batch_size, show_progress_bar=False).astype(np.float32)
        medoid_indices = select_medoid_indices(embeddings, args.units_per_client, args.seed + client_id)
        farthest_indices = select_farthest_first(embeddings, args.units_per_client)
        buckets = discriminative_candidates(docs, token_client_df, entity_client_df)

        r0, r0_emb = take_diverse_records("R0_cluster_medoids", client_id, docs, embeddings, medoid_indices, "kmeans_cluster_medoid")
        r1, r1_emb = take_diverse_records("R1_farthest_first", client_id, docs, embeddings, farthest_indices, "farthest_first_cosine_coverage")
        r2, r2_emb = take_discriminative_records("R2_cross_client_discriminative", client_id, docs, embeddings, buckets, args.units_per_client, "client_representative_cross_client_idf")

        r3: list[dict[str, Any]] = []
        r3_emb: list[np.ndarray] = []
        seen = []
        for record, vector in list(zip(r0, r0_emb))[:16] + list(zip(r2, r2_emb)):
            if len(r3) >= args.units_per_client or near_duplicate(record["unit_text"], seen):
                continue
            seen.append(record["unit_text"])
            copied = dict(record)
            copied["variant"] = "R3_hybrid_16_medoid_16_discriminative"
            copied["unit_id"] = f"R3_hybrid_16_medoid_16_discriminative:c{client_id:02d}:u{len(r3):02d}"
            copied["selection_method"] = "hybrid_medoid" if len(r3) < 16 else "hybrid_discriminative"
            r3.append(copied)
            r3_emb.append(vector)
        if len(r3) != args.units_per_client:
            raise RuntimeError(f"client {client_id}: hybrid profile has {len(r3)} rather than 32 units")

        profiles = [("R0_cluster_medoids", r0, r0_emb), ("R1_farthest_first", r1, r1_emb), ("R2_cross_client_discriminative", r2, r2_emb), ("R3_hybrid_16_medoid_16_discriminative", r3, np.asarray(r3_emb, dtype=np.float32))]
        elapsed = time.perf_counter() - started
        for variant, records, vectors in profiles:
            if len(records) != args.units_per_client:
                raise RuntimeError(f"{variant} client {client_id}: expected 32 units, got {len(records)}")
            all_units.extend(records)
            embeddings_payload[f"{variant}__client_{client_id:02d}"] = vectors
            text_bytes = sum(record["utf8_bytes"] for record in records)
            embedding_bytes = int(vectors.size * vectors.dtype.itemsize)
            stats.append({
                "dataset": args.dataset,
                "variant": variant,
                "client_id": client_id,
                "units_per_client": len(records),
                "tokens_per_client": sum(record["token_count"] for record in records),
                "text_bytes_per_client": text_bytes,
                "selected_embedding_bytes_per_client": embedding_bytes,
                # The reported profile cost includes the selected memory text and
                # its 32 stored vectors; full local document vectors are never
                # persisted or exposed by this stage.
                "bytes_per_client": text_bytes + embedding_bytes,
                "profile_construction_seconds": round(elapsed, 6),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "title_units": sum(record["unit_type"] == "title" for record in records),
                "entity_units": sum(record["unit_type"] == "entity_group" for record in records),
                "snippet_units": sum(record["unit_type"] == "snippet" for record in records),
                "relation_units": sum(record["unit_type"] == "rare_relation_phrase" for record in records),
            })
        print(json.dumps({"status": "profile_complete", "dataset": args.dataset, "client": client_id, "train_docs": len(docs), "variants": 4}), flush=True)

    write_jsonl(args.output_dir / "client_memory_units.jsonl", all_units)
    np.savez_compressed(args.output_dir / "selected_unit_embeddings.npz", **embeddings_payload)
    with (args.output_dir / "memory_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    manifest = {
        "stage": "R2-A.6_REMP",
        "dataset": args.dataset,
        "encoder": args.encoder,
        "units_per_client": args.units_per_client,
        "unit_token_limits": {"title": 16, "entity_group": 12, "snippet": 32, "rare_relation_phrase": 12},
        "duplicate_rule": {"exact_normalized": True, "near_duplicate_token_jaccard": NEAR_DUP_JACCARD},
        "minimum_unit_diversity": {"R2": ["title", "entity_group", "snippet", "rare_relation_phrase"], "R3": ["medoid", "discriminative"]},
        "seed": args.seed,
        "router_train_only": True,
        "gold_answer_support_or_development_fields_used": False,
        "full_document_embeddings_published": False,
        "selected_unit_embedding_file": str((args.output_dir / "selected_unit_embeddings.npz").resolve()),
        "reader_started": False,
        "final_test_accessed": False,
    }
    (args.output_dir / "profile_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
