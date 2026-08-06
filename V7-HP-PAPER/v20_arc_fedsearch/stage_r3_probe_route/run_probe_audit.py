#!/usr/bin/env python3
"""Stage R3 offline audit for query-time, no-body federated probes.

All client rankings and probe payloads are calculated before support labels are
read.  Gold-derived fields are attached only to offline metrics and the O2
cross-validated diagnostic upper bound.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
ENTITY = re.compile(r"(?:[A-Z][A-Za-z0-9'-]*(?:\s+|$)){1,5}")
CLIENTS = 20
DEPTH = 10
TRANSMIT_PER_CLIENT = 5
RRF_K = 60


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def document_id(dataset: str, title: str, text: str = "") -> str:
    value = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(value.encode()).hexdigest()[:20]}"


def support_docs(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {
            document_id(dataset, item.get("title", ""), item.get("paragraph_text", ""))
            for item in row.get("paragraphs", [])
            if item.get("is_supporting", item.get("is_support", False))
        }
    facts = row.get("supporting_facts", {})
    titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
    return {document_id(dataset, title) for title in titles}


def query_terms(text: str) -> set[str]:
    return {token.lower() for token in TOKEN.findall(text) if len(token) > 2}


def entities(text: str) -> set[str]:
    return {" ".join(match.split()).lower() for match in ENTITY.findall(text) if match.strip()}


def fts_query(question: str) -> str:
    terms = list(dict.fromkeys(token.lower() for token in TOKEN.findall(question) if len(token) > 1))[:32]
    return " OR ".join(f'"{term}"' for term in terms)


def sparse_search(connection: sqlite3.Connection, question: str, limit: int) -> list[dict[str, Any]]:
    query = fts_query(question)
    if not query:
        return []
    sql = """SELECT d.doc_id, d.title, d.text, -bm25(docs_fts,2.0,1.0) AS sparse_score
             FROM docs_fts JOIN docs d ON d.id=docs_fts.rowid
             WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts,2.0,1.0) LIMIT ?"""
    names = ("doc_id", "title", "text", "sparse_score")
    return [dict(zip(names, value)) for value in connection.execute(sql, (query, limit))]


def stable_ranking(values: list[float]) -> list[int]:
    return [int(index) for index in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")]


def minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(array.max() - array.min())
    return np.zeros_like(array) if span <= 1e-12 else (array - array.min()) / span


def entropy(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return 0.0
    shifted = array - array.max()
    prob = np.exp(shifted)
    prob /= prob.sum()
    return float(-(prob * np.log(prob + 1e-12)).sum() / math.log(max(2, len(prob))))


def title_diversity(titles: list[str]) -> float:
    token_sets = [query_terms(title) for title in titles]
    if len(token_sets) < 2:
        return 0.0
    similarities = []
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            similarities.append(len(left & right) / max(1, len(left | right)))
    return float(1.0 - np.mean(similarities))


def entity_diversity(titles: list[str]) -> float:
    token_sets = [entities(title) for title in titles]
    if len(token_sets) < 2:
        return 0.0
    similarities = []
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            similarities.append(len(left & right) / max(1, len(left | right)))
    return float(1.0 - np.mean(similarities))


def rank_correlation(dense: list[dict[str, Any]], sparse: list[dict[str, Any]]) -> float:
    dense_rank = {str(doc["doc_id"]): rank for rank, doc in enumerate(dense)}
    sparse_rank = {str(doc["doc_id"]): rank for rank, doc in enumerate(sparse)}
    common = sorted(set(dense_rank) & set(sparse_rank))
    if len(common) < 2:
        return 0.0
    left = np.asarray([dense_rank[key] for key in common], dtype=np.float64)
    right = np.asarray([sparse_rank[key] for key in common], dtype=np.float64)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def compact(doc: dict[str, Any], rank: int, ranker: str) -> dict[str, Any]:
    return {
        "doc_id": str(doc["doc_id"]),
        "title": str(doc["title"]),
        "client_id": int(doc["client_id"]),
        "local_rank": rank,
        "dense_score": float(doc["dense_score"]),
        "bm25_score": float(doc["sparse_score"]),
        "ranker": ranker,
        "payload_bytes": len((str(doc["title"]) + "\n" + str(doc["text"])).encode("utf-8")),
    }


def average_precision(labels: np.ndarray, values: np.ndarray) -> float:
    positives = int(labels.sum())
    if not positives:
        return 0.0
    order = np.argsort(-values, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return float((precision * ranked).sum() / positives)


def auc(labels: np.ndarray, values: np.ndarray) -> float:
    positive = values[labels == 1]
    negative = values[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def bootstrap_stat(labels: np.ndarray, values: np.ndarray, function, seed: int) -> tuple[float, float, float]:
    point = float(function(labels, values))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(1000):
        indices = rng.integers(0, len(labels), len(labels))
        sample_labels = labels[indices]
        if sample_labels.min() == sample_labels.max():
            continue
        draws.append(function(sample_labels, values[indices]))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bootstrap_delta(new: list[float], old: list[float], seed: int) -> tuple[float, float, float]:
    new_array, old_array = np.asarray(new), np.asarray(old)
    point = float(new_array.mean() - old_array.mean())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(3000):
        indices = rng.integers(0, len(new_array), len(new_array))
        draws.append(float((new_array[indices] - old_array[indices]).mean()))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def full_support(gold: set[str], documents: list[dict[str, Any]]) -> int:
    return int(bool(gold) and gold.issubset({str(doc["doc_id"]) for doc in documents}))


def merge_documents(documents: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method == "raw":
        return sorted(documents, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:10]
    by_client: dict[int, list[dict[str, Any]]] = {}
    for document in documents:
        by_client.setdefault(int(document["client_id"]), []).append(document)
    for values in by_client.values():
        values.sort(key=lambda doc: int(doc["local_rank"]))
        count = max(1, len(values))
        for document in values:
            document["percentile_score"] = 1.0 - int(document["local_rank"]) / count
    return sorted(documents, key=lambda doc: (-float(doc["percentile_score"]), -float(doc["dense_score"]), str(doc["doc_id"])))[:10]


def mean_metric(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records]))


def cross_validated_oracle(feature_rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    """Diagnostic only: five query-fold logistic regression, never deployed."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"available": False, "reason": "sklearn_missing"}
    query_ids = sorted({str(record["query_id"]) for record in feature_rows})
    fold = {query: index % 5 for index, query in enumerate(query_ids)}
    labels = np.asarray([int(record["offline_positive_client"]) for record in feature_rows])
    features = np.asarray([[float(record[name]) for name in feature_names] for record in feature_rows], dtype=np.float64)
    predictions = np.zeros(len(feature_rows), dtype=np.float64)
    for value in range(5):
        train = np.asarray([fold[str(record["query_id"])] != value for record in feature_rows])
        test = ~train
        if labels[train].min() == labels[train].max():
            continue
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260806)
        model.fit(features[train], labels[train])
        predictions[test] = model.predict_proba(features[test])[:, 1]
    # Query-wise Top-3 conversion is the interpretable bound. Ranking first,
    # then attach complete-set labels that are not available to the model.
    per_query: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for score, record in zip(predictions, feature_rows):
        per_query.setdefault(str(record["query_id"]), []).append((float(score), record))
    conversion = []
    for values in per_query.values():
        values.sort(key=lambda item: (-item[0], int(item[1]["client_id"])))
        gold = set(json.loads(values[0][1]["gold_clients_offline_only"]))
        selected = {int(item[1]["client_id"]) for item in values[:3]}
        conversion.append(int(gold.issubset(selected)))
    return {
        "available": True,
        "folds": 5,
        "client_auc": auc(labels, predictions),
        "client_auprc": average_precision(labels, predictions),
        "complete_client_set_recall_at_3": float(np.mean(conversion)),
        "gold_used_for_offline_cv_upper_bound_only": True,
        "deployable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sparse-candidates", type=int, default=100)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    data = list(rows(args.split))
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))["profiles"]
    p0 = np.asarray([profile["p0_single_centroid"] for profile in profiles], dtype=np.float32)
    assignment = {str(item["doc_id"]): int(item["client_id"]) for item in rows(args.assignment)}
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=args.device)
    connections = {client: sqlite3.connect(args.local_index_root / f"client_{client:02d}.sqlite") for client in range(CLIENTS)}

    probe_records: list[dict[str, Any]] = []
    query_cache: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    try:
        for number, row in enumerate(data, start=1):
            query = qid(row)
            question = str(row["question"])
            query_embedding = model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
            static_scores = (p0 @ query_embedding).astype(float).tolist()
            static_rank = stable_ranking(static_scores)
            candidate_rank = {client: rank for rank, client in enumerate(static_rank)}
            q_terms, q_entities = query_terms(question), entities(question)
            shallow: dict[int, dict[str, Any]] = {}
            for client, connection in connections.items():
                client_started = time.perf_counter()
                documents = sparse_search(connection, question, args.sparse_candidates)
                for document in documents:
                    document["client_id"] = client
                if documents:
                    encodings = model.encode(
                        [f"{document['title']}. {document['text']}" for document in documents],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        batch_size=256,
                        show_progress_bar=False,
                    )
                    for document, score in zip(documents, (encodings @ query_embedding).astype(float).tolist()):
                        document["dense_score"] = score
                dense = sorted(documents, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:DEPTH]
                sparse = sorted(documents, key=lambda doc: (-float(doc["sparse_score"]), str(doc["doc_id"])))[:DEPTH]
                dense_compact = [compact(document, rank, "dense") for rank, document in enumerate(dense)]
                sparse_compact = [compact(document, rank, "bm25") for rank, document in enumerate(sparse)]
                dense_scores = [float(document["dense_score"]) for document in dense]
                sparse_scores = [float(document["sparse_score"]) for document in sparse]
                dense_top = dense[0] if dense else {"title": ""}
                titles = [str(document["title"]) for document in dense[:3]]
                title_similarity = 0.0
                if dense:
                    title_embedding = model.encode(
                        [str(dense_top["title"])], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
                    )[0]
                    title_similarity = float(query_embedding @ title_embedding)
                dense_ids, sparse_ids = {str(document["doc_id"]) for document in dense[:3]}, {str(document["doc_id"]) for document in sparse[:3]}
                matched_title_tokens = q_terms & query_terms(str(dense_top["title"]))
                matched_entities = q_entities & entities(str(dense_top["title"]))
                shallow[client] = {
                    "dense": dense_compact,
                    "bm25": sparse_compact,
                    "feature": {
                        "dense_top1_score": dense_scores[0] if dense_scores else 0.0,
                        "dense_top3_mean": float(np.mean(dense_scores[:3])) if dense_scores else 0.0,
                        "dense_top1_top2_margin": (dense_scores[0] - dense_scores[1]) if len(dense_scores) > 1 else 0.0,
                        "dense_score_std": float(np.std(dense_scores)) if dense_scores else 0.0,
                        "dense_score_entropy": entropy(dense_scores),
                        # The percentile of the strongest dense result within this
                        # client's fixed depth-10 list; supplied as an explicit
                        # scalar even though dense Top-1 fixes it at 1.0.
                        "dense_local_rank_percentile": 1.0 if dense_scores else 0.0,
                        "bm25_top1_score": sparse_scores[0] if sparse_scores else 0.0,
                        "bm25_top3_mean": float(np.mean(sparse_scores[:3])) if sparse_scores else 0.0,
                        "bm25_top1_top2_margin": (sparse_scores[0] - sparse_scores[1]) if len(sparse_scores) > 1 else 0.0,
                        "dense_bm25_top1_same": int(bool(dense and sparse and dense[0]["doc_id"] == sparse[0]["doc_id"])),
                        "dense_bm25_top3_overlap": len(dense_ids & sparse_ids) / 3.0,
                        "dense_sparse_rank_correlation": rank_correlation(dense, sparse),
                        "matched_query_entity_count": len(matched_entities),
                        "matched_query_token_count": len(q_terms & query_terms(" ".join(titles))),
                        "matched_title_token_count": len(matched_title_tokens),
                        "query_title_embedding_similarity": title_similarity,
                        "top3_title_diversity": title_diversity(titles),
                        "top3_entity_diversity": entity_diversity(titles),
                    },
                    # The local retrieval is executed again by a selected client
                    # during the formal deep phase.  We retain its measured cost
                    # here without returning document content in the probe payload.
                    "local_retrieval_latency_ms": (time.perf_counter() - client_started) * 1000.0,
                }
            # Store feature/payload records before opening support annotations.
            for client in range(CLIENTS):
                feature = shallow[client]["feature"]
                probe_payload = {
                    "client_id": client,
                    "static_candidate_rank": candidate_rank[client] + 1,
                    "static_score": static_scores[client],
                    **feature,
                    "top3_title_summary": [document["title"] for document in shallow[client]["dense"][:3]],
                    "top3_entity_summary": sorted(set().union(*(entities(document["title"]) for document in shallow[client]["dense"][:3])))[:12],
                }
                probe_records.append({
                    "dataset": args.dataset,
                    "query_id": query,
                    "client_id": client,
                    **probe_payload,
                    "probe_payload_bytes": len(json.dumps(probe_payload, sort_keys=True).encode("utf-8")),
                    "local_retrieval_latency_ms": shallow[client]["local_retrieval_latency_ms"],
                    "probe_return_contains_document_text": False,
                    "probe_return_contains_full_embedding": False,
                    "probe_return_contains_gold_or_reader": False,
                })
            query_cache[query] = {
                "static_rank": static_rank,
                "static_scores": static_scores,
                "shallow": shallow,
            }
            if number % 10 == 0:
                print(json.dumps({"status": "running", "dataset": args.dataset, "completed": number, "target": len(data), "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    finally:
        for connection in connections.values():
            connection.close()

    # Labels are attached only after every P0 and local probe value is frozen.
    by_query = {qid(row): row for row in data}
    for record in probe_records:
        row = by_query[record["query_id"]]
        gold_documents = support_docs(row, args.dataset)
        gold_clients = sorted({assignment[document] for document in gold_documents if document in assignment})
        record["offline_positive_client"] = int(int(record["client_id"]) in set(gold_clients))
        record["gold_clients_offline_only"] = json.dumps(gold_clients)

    feature_names = [
        "static_score", "dense_top1_score", "dense_top3_mean", "dense_top1_top2_margin", "dense_score_std", "dense_score_entropy",
        "dense_local_rank_percentile", "bm25_top1_score", "bm25_top3_mean", "bm25_top1_top2_margin", "dense_bm25_top1_same",
        "dense_bm25_top3_overlap", "dense_sparse_rank_correlation", "matched_query_entity_count", "matched_query_token_count",
        "matched_title_token_count", "query_title_embedding_similarity", "top3_title_diversity", "top3_entity_diversity",
    ]
    discrimination = []
    candidate_records = [record for record in probe_records if int(record["static_candidate_rank"]) <= 8]
    labels = np.asarray([int(record["offline_positive_client"]) for record in candidate_records])
    for index, name in enumerate(feature_names):
        values = np.asarray([float(record[name]) for record in candidate_records], dtype=np.float64)
        auc_value, auc_low, auc_high = bootstrap_stat(labels, values, auc, 20260806 + index)
        pr_value, pr_low, pr_high = bootstrap_stat(labels, values, average_precision, 30260806 + index)
        positive, negative = values[labels == 1], values[labels == 0]
        pooled = math.sqrt((positive.var() + negative.var()) / 2.0) if len(positive) and len(negative) else 0.0
        discrimination.append({
            "dataset": args.dataset,
            "feature": name,
            "candidate_scope": "P0_top8",
            "positive_client_rows": int(labels.sum()),
            "negative_client_rows": int((1 - labels).sum()),
            "auc": auc_value,
            "auc_ci95_low": auc_low,
            "auc_ci95_high": auc_high,
            "auprc": pr_value,
            "auprc_ci95_low": pr_low,
            "auprc_ci95_high": pr_high,
            "effect_size_cohens_d": float((positive.mean() - negative.mean()) / pooled) if pooled else 0.0,
            "gold_used_for_offline_audit_only": True,
        })

    offline_oracle = cross_validated_oracle(candidate_records, feature_names)
    write_csv(output / "probe_features" / "feature_discrimination.csv", discrimination)
    with (output / "probe_features" / "per_query_client_probe.jsonl").open("w", encoding="utf-8") as handle:
        for record in probe_records:
            # Keep the deployable probe transcript label-free.  Offline labels
            # remain in memory solely for the discrimination/O2 audit files.
            transcript = {
                key: value
                for key, value in record.items()
                if key not in {"offline_positive_client", "gold_clients_offline_only"}
            }
            handle.write(json.dumps(transcript, ensure_ascii=False) + "\n")

    baselines: list[dict[str, Any]] = []
    upper: list[dict[str, Any]] = []
    for row in data:
        query = qid(row)
        cache = query_cache[query]
        gold_documents = support_docs(row, args.dataset)
        gold_clients = sorted({assignment[document] for document in gold_documents if document in assignment})
        gold_set = set(gold_clients)
        static_rank = cache["static_rank"]
        static_scores = cache["static_scores"]
        probe_rows = {int(record["client_id"]): record for record in probe_records if record["query_id"] == query}
        oracle_subset = int(len(gold_set) <= 3 and gold_set.issubset(set(static_rank[:8])))
        upper.append({
            "dataset": args.dataset,
            "query_id": query,
            "O0_static_independent_top3": int(gold_set.issubset(set(static_rank[:3]))),
            "O1_oracle_subset_at3_within_top8": oracle_subset,
            "gold_client_count_offline_only": len(gold_set),
            "gold_clients_offline_only": json.dumps(gold_clients),
            "gold_used_for_upper_bound_only": True,
        })
        for candidate_limit in (5, 8):
            candidates = static_rank[:candidate_limit]
            dense_top1 = [float(probe_rows[client]["dense_top1_score"]) for client in candidates]
            dense_top3 = [float(probe_rows[client]["dense_top3_mean"]) for client in candidates]
            percentile = [float(probe_rows[client]["dense_local_rank_percentile"]) for client in candidates]
            bm25_top1 = [float(probe_rows[client]["bm25_top1_score"]) for client in candidates]
            local_rank_dense = stable_ranking(dense_top1)
            local_rank_bm25 = stable_ranking(bm25_top1)
            rrf = [1.0 / (RRF_K + local_rank_dense[index] + 1) + 1.0 / (RRF_K + local_rank_bm25[index] + 1) for index in range(len(candidates))]
            strategies: dict[str, list[int]] = {
                "P0_static_single_centroid": static_rank[:3],
                "P1_probe_dense_top1": [candidates[index] for index in stable_ranking(dense_top1)[:3]],
                "P2_probe_dense_top3_mean": [candidates[index] for index in stable_ranking(dense_top3)[:3]],
                "P3_probe_dense_rank_percentile": [candidates[index] for index in stable_ranking(percentile)[:3]],
                "P4_probe_dense_bm25_rrf": [candidates[index] for index in stable_ranking(rrf)[:3]],
            }
            static_norm, probe_norm = minmax([static_scores[client] for client in candidates]), minmax(dense_top3)
            for alpha in (0.25, 0.50, 0.75):
                values = alpha * static_norm + (1.0 - alpha) * probe_norm
                strategies[f"P5_static_plus_probe_alpha_{alpha:.2f}"] = [candidates[index] for index in stable_ranking(values.tolist())[:3]]
            for name, selected in strategies.items():
                selected_set = set(selected)
                docs10 = [document for client in selected for document in cache["shallow"][client]["dense"][:DEPTH]]
                transmitted = [document for client in selected for document in cache["shallow"][client]["dense"][:TRANSMIT_PER_CLIENT]]
                raw = merge_documents([dict(document) for document in transmitted], "raw")
                calibrated = merge_documents([dict(document) for document in transmitted], "percentile")
                baseline_selected = set(static_rank[:3])
                omitted_gold = gold_set - baseline_selected
                baselines.append({
                    "dataset": args.dataset,
                    "query_id": query,
                    "candidate_L": candidate_limit,
                    "method": name,
                    "selected_clients": json.dumps(selected),
                    "client_coverage_at_3": int(gold_set.issubset(selected_set)),
                    "gold_client_recall_at_3": len(gold_set & selected_set) / max(1, len(gold_set)),
                    "complete_client_set_recall_at_3": int(gold_set.issubset(selected_set)),
                    "local_complete_at_10": full_support(gold_documents, docs10),
                    "transmitted_complete_at_15": full_support(gold_documents, transmitted),
                    "raw_merged_complete_at_10": full_support(gold_documents, raw),
                    "percentile_merged_complete_at_10": full_support(gold_documents, calibrated),
                    "in_p0_top8_complete_set": int(gold_set.issubset(set(static_rank[:8]))),
                    "top8_to_top3_conversion": int(gold_set.issubset(selected_set)) if gold_set.issubset(set(static_rank[:8])) else "",
                    "second_hop_client_recall": (len(gold_set & selected_set) / len(gold_set)) if len(gold_set) > 1 else "",
                    "omitted_gold_client_rescue": int(bool(omitted_gold & selected_set)),
                    "false_replacement_count": len(selected_set - gold_set) - len(baseline_selected - gold_set),
                    "probe_clients_contacted": candidate_limit,
                    "probe_bytes": sum(int(probe_rows[client]["probe_payload_bytes"]) for client in candidates),
                    "deep_clients_contacted": 3,
                    "document_bytes": sum(int(document["payload_bytes"]) for document in transmitted),
                    "probe_latency_ms": sum(float(probe_rows[client]["local_retrieval_latency_ms"]) for client in candidates),
                    "deep_retrieval_latency_ms": sum(float(probe_rows[client]["local_retrieval_latency_ms"]) for client in selected),
                    "gold_used_after_ranking_for_offline_metrics_only": True,
                    "reader_started": False,
                })

    write_csv(output / "probe_oracle" / "probe_upper_bound_per_query.csv", upper)
    o0 = [float(record["O0_static_independent_top3"]) for record in upper]
    o1 = [float(record["O1_oracle_subset_at3_within_top8"]) for record in upper]
    o1_gain, o1_low, o1_high = bootstrap_delta(o1, o0, 20260806)
    oracle_summary = [{
        "dataset": args.dataset,
        "O0_static_independent_top3": float(np.mean(o0)),
        "O1_oracle_subset_at3_within_top8": float(np.mean(o1)),
        "O1_minus_O0": o1_gain,
        "O1_minus_O0_ci95_low": o1_low,
        "O1_minus_O0_ci95_high": o1_high,
        **offline_oracle,
    }]
    write_csv(output / "probe_oracle" / "probe_upper_bound.csv", oracle_summary)
    write_csv(output / "label_free_baselines" / "per_query_results.csv", baselines)

    summary = []
    for candidate_limit in (5, 8):
        reference = [record for record in baselines if record["candidate_L"] == candidate_limit and record["method"] == "P0_static_single_centroid"]
        for method in sorted({str(record["method"]) for record in baselines if record["candidate_L"] == candidate_limit}):
            values = [record for record in baselines if record["candidate_L"] == candidate_limit and record["method"] == method]
            coverage = [float(record["complete_client_set_recall_at_3"]) for record in values]
            ref_coverage = [float(record["complete_client_set_recall_at_3"]) for record in reference]
            gain, low, high = bootstrap_delta(coverage, ref_coverage, 40260806 + candidate_limit)
            subset = [record for record in values if int(record["in_p0_top8_complete_set"]) == 1]
            summary.append({
                "dataset": args.dataset,
                "candidate_L": candidate_limit,
                "method": method,
                "queries": len(values),
                "client_coverage_at_3": mean_metric(values, "client_coverage_at_3"),
                "gold_client_recall_at_3": mean_metric(values, "gold_client_recall_at_3"),
                "complete_client_set_recall_at_3": mean_metric(values, "complete_client_set_recall_at_3"),
                "coverage_minus_P0": gain,
                "coverage_minus_P0_ci95_low": low,
                "coverage_minus_P0_ci95_high": high,
                "local_complete_at_10": mean_metric(values, "local_complete_at_10"),
                "transmitted_complete_at_15": mean_metric(values, "transmitted_complete_at_15"),
                "raw_merged_complete_at_10": mean_metric(values, "raw_merged_complete_at_10"),
                "percentile_merged_complete_at_10": mean_metric(values, "percentile_merged_complete_at_10"),
                "p0_top8_subset_queries": len(subset),
                "p0_top8_to_top3_conversion_rate": mean_metric(subset, "complete_client_set_recall_at_3") if subset else 0.0,
                "omitted_gold_client_rescue_count": int(sum(int(record["omitted_gold_client_rescue"]) for record in values)),
                "false_replacement_count": int(sum(int(record["false_replacement_count"]) for record in values)),
                "mean_probe_bytes": mean_metric(values, "probe_bytes"),
                "mean_document_bytes": mean_metric(values, "document_bytes"),
                "mean_probe_latency_ms": mean_metric(values, "probe_latency_ms"),
                "mean_deep_retrieval_latency_ms": mean_metric(values, "deep_retrieval_latency_ms"),
                "deep_clients_contacted": 3,
                "documents_transmitted": 15,
                "reader_started": False,
            })
    write_csv(output / "label_free_baselines" / "main_results.csv", summary)

    p0 = [record for record in summary if record["candidate_L"] == 8 and record["method"] == "P0_static_single_centroid"][0]
    alternatives = [record for record in summary if record["candidate_L"] == 8 and record["method"] != "P0_static_single_centroid"]
    best = max(alternatives, key=lambda record: float(record["complete_client_set_recall_at_3"]))
    status = "query_time_probe_signal_confirmed" if float(best["coverage_minus_P0"]) >= 0.05 and float(best["local_complete_at_10"]) >= float(p0["local_complete_at_10"]) + 0.03 else "query_time_probe_failed"
    decision = {
        "stage": "R3_ProbeRoute_FedRAG_Probe_Dev",
        "dataset": args.dataset,
        "status": status,
        "best_label_free_method": best["method"],
        "best_coverage_gain_vs_P0": best["coverage_minus_P0"],
        "best_coverage_gain_ci95": [best["coverage_minus_P0_ci95_low"], best["coverage_minus_P0_ci95_high"]],
        "reader_start_decision": "blocked_before_fresh_holdout",
        "final_test_accessed": False,
        "no_gold_for_probe_or_label_free_ranking": True,
    }
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "reports" / "probe_route_go_no_go.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (output / "reports" / "reader_start_decision.json").write_text(json.dumps({"status": "blocked_before_fresh_holdout", "reader_started": False}, indent=2) + "\n", encoding="utf-8")
    (output / "reports" / "next_method_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    no_leak = {
        "dataset": args.dataset,
        "probe_features_use_gold_support_answer_or_reader": False,
        "label_free_rankings_use_gold_support_answer_or_reader": False,
        "probe_returns_document_text": False,
        "probe_returns_full_document_embeddings": False,
        "reader_started": False,
        "final_test_accessed": False,
        "o2_is_offline_cross_validated_upper_bound_only": True,
    }
    (output / "protocol_no_leak_audit.json").write_text(json.dumps(no_leak, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
