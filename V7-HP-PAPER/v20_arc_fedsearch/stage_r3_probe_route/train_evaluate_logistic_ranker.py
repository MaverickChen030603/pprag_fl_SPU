#!/usr/bin/env python3
"""Train/evaluate the frozen R3 logistic client ranker without a reader."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_compact_payload_audit import FEATURE_SCHEMA
from run_probe_audit import support_docs


SEEDS = (20260807, 20260808, 20260809)
RRF_K = 60
WIRE_BYTES_L8 = 592


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def stable_ranking(values: list[float]) -> list[int]:
    return [int(index) for index in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")]


def minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(array.max() - array.min())
    return np.zeros_like(array) if span <= 1e-12 else (array - array.min()) / span


def feature_vector(record: dict[str, Any]) -> list[float]:
    return [float(record["static_score"])] + [float(record[name]) for name in FEATURE_SCHEMA]


def labels_for_split(split: Path, assignment: dict[str, int], dataset: str) -> dict[str, set[int]]:
    values = {}
    for row in rows(split):
        values[qid(row)] = {assignment[document] for document in support_docs(row, dataset) if document in assignment}
    return values


def selected_documents(packet: dict[str, Any], selected: list[int]) -> list[dict[str, Any]]:
    docs = packet["local_dense_docs_top10"]
    return [document for client in selected for document in docs[str(client)][:5]]


def selection_label_free(records: list[dict[str, Any]], dataset: str) -> list[int]:
    ordered = sorted(records, key=lambda record: int(record["static_candidate_rank"]))
    if dataset == "musique":
        return [int(records[index]["client_id"]) for index in stable_ranking([float(record["dense_top1_score"]) for record in records])[:3]]
    static_norm = minmax([float(record["static_score"]) for record in records])
    probe_norm = minmax([float(record["dense_top3_mean"]) for record in records])
    score = 0.25 * static_norm + 0.75 * probe_norm
    return [int(records[index]["client_id"]) for index in stable_ranking(score.tolist())[:3]]


def local_complete(gold: set[str], packet: dict[str, Any], selected: list[int]) -> int:
    docs = packet["local_dense_docs_top10"]
    available = {str(document["doc_id"]) for client in selected for document in docs[str(client)][:10]}
    return int(bool(gold) and gold.issubset(available))


def evaluate_one(packet: dict[str, Any], selected: list[int], gold_docs: set[str], gold_clients: set[int]) -> dict[str, Any]:
    transmitted = selected_documents(packet, selected)
    raw = sorted(transmitted, key=lambda document: (-float(document["dense_score"]), str(document["doc_id"])))[:10]
    percentile = sorted(transmitted, key=lambda document: (int(document["local_rank"]), -float(document["dense_score"]), str(document["doc_id"])))[:10]
    complete = lambda values: int(bool(gold_docs) and gold_docs.issubset({str(document["doc_id"]) for document in values}))
    return {
        "selected_clients": json.dumps(selected),
        "complete_client_set_recall_at_3": int(gold_clients.issubset(set(selected))),
        "gold_client_recall_at_3": len(gold_clients & set(selected)) / max(1, len(gold_clients)),
        "local_complete_at_10": local_complete(gold_docs, packet, selected),
        "transmitted_complete_at_15": complete(transmitted),
        "raw_merged_complete_at_10": complete(raw),
        "percentile_merged_complete_at_10": complete(percentile),
        "document_bytes": sum(int(document["payload_bytes"]) for document in transmitted),
    }


def train(args: argparse.Namespace) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment)}
    gold = labels_for_split(args.split, assignment, args.dataset)
    examples, labels, query_ids = [], [], []
    for packet in rows(args.packets):
        query = str(packet["query_id"])
        for record in packet["p0_candidate_records"]:
            examples.append(feature_vector(record))
            labels.append(int(int(record["client_id"]) in gold[query]))
            query_ids.append(query)
    features, target = np.asarray(examples, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if target.min() == target.max():
        raise ValueError("Probe-Train has one class only")
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for seed in SEEDS:
        model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed)
        model.fit(scaled, target)
        probabilities = model.predict_proba(scaled)[:, 1]
        payload = {"model": model, "scaler": scaler, "feature_names": ["static_score", *FEATURE_SCHEMA], "seed": seed}
        with (args.output_dir / f"logistic_seed_{seed}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        metrics.append({
            "dataset": args.dataset,
            "seed": seed,
            "train_queries": len(set(query_ids)),
            "candidate_rows": len(target),
            "positive_rows": int(target.sum()),
            "hard_negative_rows": int((1 - target).sum()),
            "train_auc": float(roc_auc_score(target, probabilities)),
            "train_auprc": float(average_precision_score(target, probabilities)),
            "model": "class_balanced_logistic_regression",
            "reader_started": False,
        })
    write_csv(args.output_dir / "model_results.csv", metrics)
    (args.output_dir / "training_manifest.json").write_text(json.dumps({
        "dataset": args.dataset, "candidate_L": 8, "seeds": list(SEEDS), "feature_names": ["static_score", *FEATURE_SCHEMA],
        "negative_sampling": "all_P0_top8_non_support_clients_only", "reader_started": False, "final_test_accessed": False,
    }, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment)}
    split_rows = {qid(row): row for row in rows(args.split)}
    per_query, summary = [], []
    model_payloads = []
    for seed in SEEDS:
        with (args.models_dir / f"logistic_seed_{seed}.pkl").open("rb") as handle:
            model_payloads.append((seed, pickle.load(handle)))
    for packet in rows(args.packets):
        query = str(packet["query_id"])
        records = packet["p0_candidate_records"]
        p0 = [int(record["client_id"]) for record in sorted(records, key=lambda record: int(record["static_candidate_rank"]))[:3]]
        strategies = {"B1_static_p0": p0, "B3_label_free_probe": selection_label_free(records, args.dataset)}
        inherited = packet.get("inherited_selected_clients")
        if inherited:
            strategies["B0_inherited_route"] = [int(client) for client in inherited]
        feature_matrix = np.asarray([feature_vector(record) for record in records], dtype=np.float64)
        for seed, payload in model_payloads:
            probabilities = payload["model"].predict_proba(payload["scaler"].transform(feature_matrix))[:, 1]
            strategies[f"B4_logistic_seed_{seed}"] = [int(records[index]["client_id"]) for index in stable_ranking(probabilities.tolist())[:3]]
        # Rankings above use only static scores and the frozen label-free probe
        # packet. The holdout support labels are accessed only for final metrics.
        current_gold_docs = support_docs(split_rows[query], args.dataset)
        current_gold_clients = {assignment[document] for document in current_gold_docs if document in assignment}
        for method, selected in strategies.items():
            result = evaluate_one(packet, selected, current_gold_docs, current_gold_clients)
            result.update({
                "dataset": args.dataset, "query_id": query, "method": method,
                "probe_clients_contacted": 8 if method.startswith(("B3", "B4")) else 0,
                "probe_bytes": WIRE_BYTES_L8 if method.startswith(("B3", "B4")) else 0,
                "deep_clients_contacted": 3, "documents_transmitted": 15,
                "reader_started": False, "final_test_accessed": False,
            })
            per_query.append(result)
    metrics = ("complete_client_set_recall_at_3", "gold_client_recall_at_3", "local_complete_at_10", "transmitted_complete_at_15", "raw_merged_complete_at_10", "percentile_merged_complete_at_10", "document_bytes")
    for method in sorted({row["method"] for row in per_query}):
        values = [row for row in per_query if row["method"] == method]
        summary.append({"dataset": args.dataset, "method": method, "queries": len(values), **{name: float(np.mean([float(row[name]) for row in values])) for name in metrics}, "probe_clients_contacted": values[0]["probe_clients_contacted"], "probe_bytes": values[0]["probe_bytes"], "deep_clients_contacted": 3, "documents_transmitted": 15, "reader_started": False})
    write_csv(args.output_dir / "per_query_results.csv", per_query)
    write_csv(args.output_dir / "main_results.csv", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        if args.models_dir is None:
            parser.error("--models-dir is required for evaluate")
        evaluate(args)


if __name__ == "__main__":
    main()
