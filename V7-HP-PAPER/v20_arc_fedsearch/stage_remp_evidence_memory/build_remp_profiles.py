#!/usr/bin/env python3
"""Build REM-P bounded representative evidence memory profiles.

The builder reads only client-local SQLite shards. It does not read queries,
support labels, answers, reader outputs, or final-test assets.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
CAP = re.compile(r"(?:[A-Z][A-Za-z0-9'-]*(?:\s+|$)){1,6}")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def terms(text: str) -> list[str]:
    return [x.lower() for x in TOKEN.findall(text) if len(x) > 2]


def entities(text: str) -> list[str]:
    out = []
    for match in CAP.findall(text):
        value = " ".join(match.split())
        if len(value) > 2:
            out.append(value)
    return out


def doc_key(title: str, text: str) -> str:
    payload = " ".join(title.lower().split()) + "\n" + " ".join(text.lower().split())
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def read_docs(path: Path) -> list[Document]:
    con = sqlite3.connect(path)
    try:
        cols = [row[1] for row in con.execute("pragma table_info(docs)").fetchall()]
        if "doc_id" in cols:
            rows = con.execute("select doc_id,title,text from docs").fetchall()
            return [Document(str(doc_id), str(title), str(text)) for doc_id, title, text in rows]
        rows = con.execute("select title,text from docs").fetchall()
        return [Document(doc_key(str(title), str(text)), str(title), str(text)) for title, text in rows]
    finally:
        con.close()


def unit_text(doc: Document, max_chars: int) -> str:
    body = " ".join(doc.text.split())
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0]
    return f"{doc.title}. {body}".strip()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def greedy_diverse_indices(emb: np.ndarray, scores: np.ndarray, count: int) -> list[int]:
    if len(emb) == 0 or count <= 0:
        return []
    selected = [int(np.argmax(scores))]
    min_dist = 1.0 - (emb @ emb[selected[0]])
    while len(selected) < min(count, len(emb)):
        combined = 0.65 * min_dist + 0.35 * (scores / max(float(scores.max()), 1e-9))
        combined[selected] = -1.0
        nxt = int(np.argmax(combined))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - (emb @ emb[nxt]))
    return selected


def make_units(
    docs: list[Document],
    doc_embeddings: np.ndarray,
    units_per_client: int,
    snippet_chars: int,
) -> list[dict[str, Any]]:
    token_df = collections.Counter(t for doc in docs for t in set(terms(doc.title + " " + doc.text)))
    entity_tf = collections.Counter(e for doc in docs for e in entities(doc.title))
    n_docs = max(1, len(docs))

    def rarity_score(doc: Document) -> float:
        toks = set(terms(doc.title + " " + doc.text))
        rare = sum(math.log((1 + n_docs) / (1 + token_df[t])) for t in toks)
        return rare + 0.2 * len(entities(doc.title))

    scores = np.asarray([rarity_score(doc) for doc in docs], dtype=np.float32)
    rare_quota = max(8, units_per_client // 3)
    entity_quota = max(8, units_per_client // 4)
    dense_quota = units_per_client

    chosen: dict[str, dict[str, Any]] = {}

    for rank, idx in enumerate(np.argsort(-scores)[:rare_quota]):
        doc = docs[int(idx)]
        chosen.setdefault(
            doc.doc_id,
            {
                "doc": doc,
                "unit_type": "rare_snippet",
                "selection_reason": "high rare-token/entity score",
                "selection_score": float(scores[int(idx)]),
                "rank": int(rank),
            },
        )

    entity_docs = []
    seen_entities = set()
    for entity, _freq in entity_tf.most_common(entity_quota * 4):
        if entity.lower() in seen_entities:
            continue
        seen_entities.add(entity.lower())
        matches = [i for i, doc in enumerate(docs) if entity in entities(doc.title)]
        if matches:
            idx = max(matches, key=lambda i: scores[i])
            entity_docs.append(idx)
        if len(entity_docs) >= entity_quota:
            break
    for rank, idx in enumerate(entity_docs):
        doc = docs[int(idx)]
        chosen.setdefault(
            doc.doc_id,
            {
                "doc": doc,
                "unit_type": "entity_anchor",
                "selection_reason": "representative title/entity anchor",
                "selection_score": float(scores[int(idx)]),
                "rank": int(rank),
            },
        )

    dense_indices = greedy_diverse_indices(doc_embeddings, scores, dense_quota)
    for rank, idx in enumerate(dense_indices):
        doc = docs[int(idx)]
        chosen.setdefault(
            doc.doc_id,
            {
                "doc": doc,
                "unit_type": "diverse_dense",
                "selection_reason": "embedding-diverse representative document",
                "selection_score": float(scores[int(idx)]),
                "rank": int(rank),
            },
        )
        if len(chosen) >= units_per_client:
            break

    units = []
    for ordinal, item in enumerate(list(chosen.values())[:units_per_client]):
        doc = item["doc"]
        units.append(
            {
                "unit_id": f"u{ordinal:04d}",
                "unit_type": item["unit_type"],
                "source_doc_id": doc.doc_id,
                "title": doc.title,
                "text": unit_text(doc, snippet_chars),
                "selection_reason": item["selection_reason"],
                "selection_score": item["selection_score"],
                "selection_rank": item["rank"],
                "entities": entities(doc.title)[:8],
            }
        )
    return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p0-centroids", type=Path)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--units-per-client", type=int, default=128)
    parser.add_argument("--snippet-chars", type=int, default=420)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    rng = np.random.default_rng(args.seed)
    _ = rng  # Keeps the seed explicit in the manifest; selection is deterministic.
    model = SentenceTransformer(args.encoder, device=args.device)
    p0 = np.load(args.p0_centroids) if args.p0_centroids else None

    profiles = []
    stats = []
    for client in range(args.clients):
        shard = args.local_index_root / f"client_{client:02d}.sqlite"
        docs = read_docs(shard)
        doc_texts = [unit_text(doc, args.snippet_chars) for doc in docs]
        doc_emb = model.encode(
            doc_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        ).astype("float32")
        if p0 is not None:
            centroid = np.asarray(p0[client], dtype=np.float32)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
        else:
            centroid = normalize_rows(doc_emb.mean(axis=0, keepdims=True))[0]

        units = make_units(docs, doc_emb, args.units_per_client, args.snippet_chars)
        unit_lookup = {doc.doc_id: i for i, doc in enumerate(docs)}
        for unit in units:
            unit["embedding"] = doc_emb[unit_lookup[unit["source_doc_id"]]].astype(float).tolist()

        tf = collections.Counter(t for doc in docs for t in terms(doc.title + " " + doc.text))
        ef = collections.Counter(e for doc in docs for e in entities(doc.title))
        profiles.append(
            {
                "dataset": args.dataset,
                "client_id": client,
                "collection_size": len(docs),
                "p0_single_centroid": centroid.astype(float).tolist(),
                "representative_units": units,
                "lexical_memory": {
                    "top_terms": [x for x, _ in tf.most_common(3000)],
                    "term_counts": dict(tf.most_common(3000)),
                    "entity_frequency_sketch": dict(ef.most_common(800)),
                },
            }
        )
        stats.append(
            {
                "dataset": args.dataset,
                "client_id": client,
                "docs": len(docs),
                "representative_units": len(units),
                "rare_units": sum(1 for u in units if u["unit_type"] == "rare_snippet"),
                "entity_units": sum(1 for u in units if u["unit_type"] == "entity_anchor"),
                "diverse_units": sum(1 for u in units if u["unit_type"] == "diverse_dense"),
            }
        )
        print(json.dumps({"status": "remp_profile_complete", "client": client, "docs": len(docs), "units": len(units)}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "REM-P",
        "dataset": args.dataset,
        "seed": args.seed,
        "encoder": args.encoder,
        "units_per_client": args.units_per_client,
        "snippet_chars": args.snippet_chars,
        "profiles": profiles,
        "gold_or_development_fields_used": False,
        "reader_started": False,
    }
    (args.output_dir / "remp_client_profiles.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    import csv

    with (args.output_dir / "remp_profile_statistics.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    manifest = {
        "status": "complete",
        "stage": "REM-P",
        "dataset": args.dataset,
        "clients": args.clients,
        "local_index_root": str(args.local_index_root.resolve()),
        "local_index_manifest_available": (args.local_index_root / "manifest.json").exists(),
        "p0_centroids": str(args.p0_centroids.resolve()) if args.p0_centroids else None,
        "p0_centroids_sha256": sha256(args.p0_centroids) if args.p0_centroids else None,
        "profile_path": str((args.output_dir / "remp_client_profiles.json").resolve()),
        "gold_or_development_fields_used": False,
        "reader_started": False,
    }
    (args.output_dir / "remp_profile_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

