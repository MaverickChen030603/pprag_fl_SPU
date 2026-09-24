#!/usr/bin/env python3
"""One-shot, label-free C1 routing and context materialization.

The program only consumes frozen blind inputs, frozen probe packets, frozen
models, and public retrieval indexes.  It intentionally contains no imports
from evaluation code and no gold-derived fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SEEDS = tuple(range(20))
FEATURES = (
    "dense_top1_score", "dense_top3_mean", "dense_top1_top2_margin", "dense_score_std",
    "dense_score_entropy", "dense_local_rank_percentile", "bm25_top1_score", "bm25_top3_mean",
    "bm25_top1_top2_margin", "dense_bm25_top1_same", "dense_bm25_top3_overlap",
    "dense_sparse_rank_correlation", "matched_query_entity_count", "matched_query_token_count",
    "matched_title_token_count", "query_title_embedding_similarity", "top3_title_diversity",
    "top3_entity_diversity",
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def static(records: list[dict[str, Any]]) -> list[int]:
    return [int(row["client_id"]) for row in sorted(records, key=lambda x: (int(x["static_candidate_rank"]), -float(x["static_score"]), int(x["client_id"])))[:3]]


def random_route(records: list[dict[str, Any]], query_id: str, seed: int) -> list[int]:
    return [int(row["client_id"]) for row in sorted(records, key=lambda x: (hashlib.sha256(f"{seed}|{query_id}|{x['client_id']}".encode()).digest(), int(x["client_id"])))[:3]]


def encoder(model: str, revision: str, device: str):
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True), AutoModel.from_pretrained(model, revision=revision, local_files_only=True).to(device).eval()


def encode(tokenizer: Any, model: Any, questions: list[str], device: str) -> np.ndarray:
    import torch
    values = []
    for start in range(0, len(questions), 64):
        batch = tokenizer(questions[start:start + 64], padding=True, truncation=True, max_length=512, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            value = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0], p=2, dim=1)
        values.append(value.float().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def merge(packet: dict[str, Any], clients: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [dict(doc) for client in clients for doc in packet["local_dense_docs_top10"][str(client)][:5]]
    expected = 5 * len(clients)
    if len(documents) != expected:
        raise ValueError(f"{packet['query_id']}: expected {expected} documents, got {len(documents)}")
    return documents, sorted(documents, key=lambda x: (-float(x["dense_score"]), str(x["doc_id"])))[:10]


def lookup(connection: sqlite3.Connection, ids: list[str]) -> list[dict[str, str]]:
    output = []
    for doc_id in ids:
        row = connection.execute("SELECT doc_id,title,text FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(f"missing canonical document {doc_id}")
        output.append({"doc_id": str(row[0]), "title": str(row[1]), "text": str(row[2])})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.stage / "retrieval/blind_retrieval_outputs.jsonl"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_name, revision = "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    tokenizer, bge = encoder(model_name, revision, args.device)
    summaries: dict[str, Any] = {}
    with output.open("x", encoding="utf-8") as handle:
        for dataset in DATASETS:
            blind = list(rows(args.stage / f"inputs/{dataset}_c1_blind_n500.jsonl"))
            packets = {str(row["query_id"]): row for row in rows(args.stage / f"retrieval/{dataset}_probe_packets.jsonl")}
            if len(blind) != 500 or set(str(row["query_id"]) for row in blind) != set(packets):
                raise ValueError(f"{dataset}: blind input and packet IDs do not match exactly")
            questions = [str(row["question"]) for row in blind]
            embeddings = encode(tokenizer, bge, questions, args.device)
            b3_centroids = np.load(args.base / f"runs/ragroute_b3_r5_posthoc_20260916/centroids/{dataset}/source_centroids.npy").astype(np.float32)
            b6_scores = embeddings @ b3_centroids.T
            with (args.base / f"runs/ragroute_b3_r5_posthoc_20260916/routes/{dataset}/ragroute_mlp.pkl").open("rb") as file:
                b3 = pickle.load(file)
            clients = b3_centroids.shape[0]
            b3_x = np.concatenate((np.repeat(embeddings, clients, axis=0), np.tile(b3_centroids, (len(blind), 1)), np.tile(np.eye(clients, dtype=np.float32), (len(blind), 1))), axis=1)
            b3_scores = b3.predict_proba(b3_x)[:, 1].reshape(len(blind), clients)
            with (args.base / f"inputs/models/{dataset}/logistic_seed_20260807.pkl").open("rb") as file:
                m2 = pickle.load(file)
            index = sqlite3.connect(f"file:{args.base / f'inputs/indexes/{dataset}.sqlite'}?mode=ro", uri=True)
            count = 0
            try:
                for position, source in enumerate(blind):
                    query_id, question = str(source["query_id"]), str(source["question"])
                    packet = packets[query_id]
                    records = list(packet["p0_candidate_records"])
                    candidates = [int(row["client_id"]) for row in records]
                    if len(candidates) != 8 or len(set(candidates)) != 8:
                        raise ValueError(f"{dataset}/{query_id}: invalid frozen Top-8")
                    matrix = np.asarray([[float(row["static_score"]), *[float(row[name]) for name in FEATURES]] for row in records])
                    m2_scores = m2["model"].predict_proba(m2["scaler"].transform(matrix))[:, 1]
                    methods: dict[str, list[int]] = {"b0_static_top3": static(records), "m2_logistic_proberoute": [int(records[i]["client_id"]) for i in np.argsort(-m2_scores, kind="stable")[:3]], "b3_ragroute_style_mlp": sorted(candidates, key=lambda c: (-float(b3_scores[position, c]), c))[:3], "b6_dense_centroid_top3": sorted(candidates, key=lambda c: (-float(b6_scores[position, c]), c))[:3], "b4a_all_candidate_top8_high_cost_reference": candidates}
                    methods.update({f"b1_random_top3_seed_{seed:02d}": random_route(records, query_id, seed) for seed in SEEDS})
                    for method, selected in methods.items():
                        if method.startswith("b4a"):
                            if selected != candidates: raise ValueError("B4a must return the frozen Top-8")
                        elif len(selected) != 3 or not set(selected).issubset(candidates):
                            raise ValueError(f"{dataset}/{query_id}/{method}: fixed budget contract violation")
                        transmitted, top10 = merge(packet, selected)
                        documents = lookup(index, [str(row["doc_id"]) for row in top10[:5]])
                        packet_hash = digest(packet)
                        context_hash = digest({"question": question, "docs": documents})
                        payload = {"dataset": dataset, "query_id": query_id, "question": question, "method": method, "candidate_client_ids": candidates, "selected_client_ids": selected, "local_doc_ids": {str(client): [str(doc["doc_id"]) for doc in packet["local_dense_docs_top10"][str(client)][:10]] for client in selected}, "transmitted_doc_ids": [str(doc["doc_id"]) for doc in transmitted], "merged_top10_doc_ids": [str(doc["doc_id"]) for doc in top10], "reader_top5_doc_ids": [str(doc["doc_id"]) for doc in top10[:5]], "reader_context_docs": documents, "probe_packet_hash": packet_hash, "context_hash": context_hash, "retrieval_output_hash": digest({"method": method, "clients": selected, "top10": [str(doc["doc_id"]) for doc in top10]}), "candidate_clients": 8, "client_budget": len(selected), "local_depth": 10, "documents_per_client": 5, "transmitted_documents": len(transmitted), "gold_or_answer_used": False, "reader_started": False}
                        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        count += 1
            finally:
                index.close()
            summaries[dataset] = {"queries": len(blind), "rows": count, "methods": 25}
    expected = 3 * 500 * 25
    actual = sum(1 for _ in rows(output))
    if actual != expected:
        raise ValueError(f"expected {expected} rows, got {actual}")
    manifest = {"status": "complete_blind_retrieval", "labels_loaded": False, "metrics_computed": False, "rows": actual, "output_sha256": sha256(output), "datasets": summaries, "methods": ["b0_static_top3", "b1_random_top3_seed_00..19", "b3_ragroute_style_mlp", "b6_dense_centroid_top3", "m2_logistic_proberoute", "b4a_all_candidate_top8_high_cost_reference"]}
    (args.stage / "retrieval/blind_retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
