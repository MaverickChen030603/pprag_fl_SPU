#!/usr/bin/env python3
"""Materialize immutable R4 reader inputs from frozen R3 artifacts.

This script never reads answers or support labels.  It reproduces the three
federated choices from stored R3 packets, fetches their already-transmitted
documents from the read-only canonical index, and retains the first five of
the raw merged Top-10 for the legacy reader contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "stage_r3_probe_route"
V16_EVAL = ROOT.parent / "v16_action_composition" / "evaluation"
sys.path.insert(0, str(V16_EVAL))
from eval_common import document_id  # noqa: E402


DATASETS = {
    "2wikimultihopqa": {
        "split": R3 / "protocol/2wikimultihopqa/probe_holdout.jsonl",
        "packets": R3 / "ranker_training/packets/2wikimultihopqa/probe_holdout.jsonl",
        "models": R3 / "ranker_training/models/2wikimultihopqa",
        "index": ROOT.parent / "v15_robust_context_repair/retrieval/indexes/2wikimultihopqa.sqlite",
        "published": R3 / "holdout/2wikimultihopqa/main_results/per_query_results.csv",
        "baseline": "B1_static_p0",
        "label_free": "B3_label_free_probe",
        "logistic": "B4_logistic_seed_20260807",
    },
    "musique": {
        "split": R3 / "protocol/musique/probe_holdout.jsonl",
        "packets": R3 / "ranker_training/packets/musique/probe_holdout.jsonl",
        "models": R3 / "ranker_training/models/musique",
        "index": ROOT.parent / "v16_action_composition/retrieval/indexes/musique.sqlite",
        "published": R3 / "holdout/musique/main_results/per_query_results.csv",
        "baseline": "B1_static_p0",
        "label_free": "B3_label_free_probe",
        "logistic": "B4_logistic_seed_20260807",
    },
    "hotpotqa": {
        "split": R3 / "hotpot_transfer/protocol/probe_holdout.jsonl",
        "packets": R3 / "hotpot_transfer/packets/probe_holdout.jsonl",
        "models": R3 / "hotpot_transfer/models",
        "index": ROOT.parent / "v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite",
        "published": R3 / "hotpot_transfer/holdout/main_results/per_query_results.csv",
        "baseline": "B0_inherited_route",
        "label_free": "B3_label_free_probe",
        "logistic": "B4_logistic_seed_20260807",
    },
}

FEATURE_SCHEMA = (
    "dense_top1_score", "dense_top3_mean", "dense_top1_top2_margin",
    "dense_score_std", "dense_score_entropy", "dense_local_rank_percentile",
    "bm25_top1_score", "bm25_top3_mean", "bm25_top1_top2_margin",
    "dense_bm25_top1_same", "dense_bm25_top3_overlap",
    "dense_sparse_rank_correlation", "matched_query_entity_count",
    "matched_query_token_count", "matched_title_token_count",
    "query_title_embedding_similarity", "top3_title_diversity", "top3_entity_diversity",
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(values: list[float]) -> list[int]:
    return [int(i) for i in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")]


def minmax(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = float(values.max() - values.min())
    return np.zeros_like(values) if span <= 1e-12 else (values - values.min()) / span


def label_free(records: list[dict[str, Any]], dataset: str) -> list[int]:
    if dataset == "musique":
        order = stable_rank([float(record["dense_top1_score"]) for record in records])
    else:
        score = .25 * minmax([float(r["static_score"]) for r in records])
        score += .75 * minmax([float(r["dense_top3_mean"]) for r in records])
        order = stable_rank(score.tolist())
    return [int(records[index]["client_id"]) for index in order[:3]]


def feature_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(record["static_score"]), *[float(record[k]) for k in FEATURE_SCHEMA]] for record in records],
        dtype=np.float64,
    )


def logistic_select(records: list[dict[str, Any]], model_path: Path) -> list[int]:
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    probabilities = payload["model"].predict_proba(payload["scaler"].transform(feature_matrix(records)))[:, 1]
    return [int(records[index]["client_id"]) for index in stable_rank(probabilities.tolist())[:3]]


def raw_merge(packet: dict[str, Any], selected: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = packet["local_dense_docs_top10"]
    transmitted = [doc for client in selected for doc in available[str(client)][:5]]
    merged = sorted(transmitted, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:10]
    return transmitted, merged


def lookup_docs(connection: sqlite3.Connection, ids: list[str]) -> list[dict[str, str]]:
    output = []
    for doc_id in ids:
        row = connection.execute("SELECT doc_id,title,text FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(f"document absent from canonical frozen index: {doc_id}")
        output.append({"doc_id": str(row[0]), "title": str(row[1]), "text": str(row[2])})
    return output


def context_hash(question: str, docs: list[dict[str, str]]) -> str:
    payload = {"question": question, "docs": docs}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_docs(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {document_id(dataset, str(p.get("title", "")), str(p.get("paragraph_text", "")))
                for p in row.get("paragraphs", []) if p.get("is_supporting", False)}
    facts = row.get("supporting_facts", {})
    titles = facts.get("title", []) if isinstance(facts, dict) else [x[0] for x in facts]
    return {document_id(dataset, str(title)) for title in titles}


def materialize_federated(dataset: str, split: dict[str, dict[str, Any]], packets: dict[str, dict[str, Any]],
                          config: dict[str, Any], connection: sqlite3.Connection) -> list[dict[str, Any]]:
    model_path = config["models"] / "logistic_seed_20260807.pkl"
    result: list[dict[str, Any]] = []
    for query_id in sorted(packets):
        packet = packets[query_id]
        records = packet["p0_candidate_records"]
        baseline = packet.get("inherited_selected_clients") if dataset == "hotpotqa" else None
        if baseline is None:
            baseline = [int(r["client_id"]) for r in sorted(records, key=lambda r: int(r["static_candidate_rank"]))[:3]]
        strategies = {
            "federated_baseline": [int(v) for v in baseline],
            "label_free_proberoute": label_free(records, dataset),
            "logistic_proberoute": logistic_select(records, model_path),
        }
        question = str(split[query_id]["question"])
        for method, selected in strategies.items():
            transmitted, merged = raw_merge(packet, selected)
            retrieved_ids = [str(doc["doc_id"]) for doc in merged]
            reader_ids = retrieved_ids[:5]
            docs = lookup_docs(connection, reader_ids)
            result.append({
                "dataset": dataset, "query_id": query_id, "question": question, "method": method,
                "retrieval_method": {"federated_baseline": config["baseline"], "label_free_proberoute": config["label_free"], "logistic_proberoute": config["logistic"]}[method],
                "selected_clients": selected, "transmitted_doc_ids": [str(d["doc_id"]) for d in transmitted],
                "retrieved_doc_ids": retrieved_ids, "reader_context_doc_ids": reader_ids,
                "reader_context_docs": docs, "context_hash": context_hash(question, docs),
                "global_pool_size": 10, "reader_context_k": 5, "source": "frozen_r3_packet_raw_merge",
                "gold_or_answer_used": False, "reader_started": False,
            })
    return result


def materialize_centralized(dataset: str, split: dict[str, dict[str, Any]], pool: Path) -> list[dict[str, Any]]:
    by_query = {str(row["query_id"]): row for row in rows(pool)}
    if set(by_query) != set(split):
        raise ValueError(f"central reference query set mismatch for {dataset}: {len(by_query)} vs {len(split)}")
    result = []
    for query_id in sorted(split):
        row = by_query[query_id]
        pool_docs = row.get("pool", [])[:10]
        ids = [str(doc["doc_id"]) for doc in pool_docs]
        docs = [{"doc_id": str(doc["doc_id"]), "title": str(doc["title"]), "text": str(doc["text"])} for doc in pool_docs[:5]]
        question = str(split[query_id]["question"])
        result.append({
            "dataset": dataset, "query_id": query_id, "question": question, "method": "centralized_retrieval_reference",
            "retrieval_method": "v17_frozen_centralized_global_hybrid_replay", "selected_clients": [0],
            "transmitted_doc_ids": ids, "retrieved_doc_ids": ids, "reader_context_doc_ids": ids[:5],
            "reader_context_docs": docs, "context_hash": context_hash(question, docs),
            "global_pool_size": 10, "reader_context_k": 5, "source": str(pool),
            "gold_or_answer_used": False, "reader_started": False,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--central-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DATASETS[args.dataset]
    split = {qid(row): row for row in rows(config["split"])}
    packets = {str(row["query_id"]): row for row in rows(config["packets"])}
    if set(split) != set(packets):
        raise ValueError(f"R3 split/packet mismatch: {len(split)} versus {len(packets)}")
    connection = sqlite3.connect(f"file:{config['index']}?mode=ro", uri=True)
    try:
        materialized = materialize_federated(args.dataset, split, packets, config, connection)
    finally:
        connection.close()
    materialized.extend(materialize_centralized(args.dataset, split, args.central_pool))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": args.dataset, "expected_queries": len(split), "rows": len(materialized), "methods": 4,
        "r3_packet_sha256": sha256(config["packets"]), "split_sha256": sha256(config["split"]),
        "central_pool_sha256": sha256(args.central_pool), "canonical_index": str(config["index"]),
        "canonical_index_sha256": sha256(config["index"]), "reader_context_k": 5,
        "raw_global_top10_then_first5": True, "gold_or_answer_used": False, "reader_started": False,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
