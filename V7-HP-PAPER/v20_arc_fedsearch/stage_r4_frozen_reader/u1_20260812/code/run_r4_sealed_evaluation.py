#!/usr/bin/env python3
"""Sealed R4 evaluator. Gold is opened only after all eight cells validate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

READERS = ("flan_t5_large", "unifiedqa_t5_large")
METHODS = ("inherited", "label_free", "logistic", "centralized")
COMPARISONS = (
    ("logistic", "inherited"),
    ("label_free", "inherited"),
    ("logistic", "label_free"),
    ("centralized", "inherited"),
)
METRICS = ("answer_em", "answer_f1", "support_em", "support_f1", "joint_em", "joint_f1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_all_cells(predictions_root: Path, contexts_root: Path, protocol_path: Path) -> dict:
    protocol_hash = sha256(protocol_path)
    records = {}
    query_order = None
    for reader in READERS:
        for method in METHODS:
            prediction_path = predictions_root / reader / f"{method}.jsonl"
            marker_path = prediction_path.with_suffix(".completed.json")
            context_path = contexts_root / f"{method}.jsonl"
            for required in (prediction_path, marker_path, context_path):
                if not required.is_file():
                    raise FileNotFoundError(required)
            marker = json.loads(marker_path.read_text())
            if marker.get("status") != "complete" or marker.get("n") != 300:
                raise ValueError(f"incomplete marker: {reader}/{method}")
            if marker.get("prediction_file_sha256") != sha256(prediction_path):
                raise ValueError(f"prediction checksum mismatch: {reader}/{method}")
            if marker.get("protocol_hash") != protocol_hash:
                raise ValueError(f"protocol checksum mismatch: {reader}/{method}")
            if marker.get("context_checksum") != sha256(context_path):
                raise ValueError(f"context checksum mismatch: {reader}/{method}")
            cell = rows(prediction_path)
            if len(cell) != 300:
                raise ValueError(f"prediction count mismatch: {reader}/{method}")
            ids = [str(row["query_id"]) for row in cell]
            if len(set(ids)) != 300:
                raise ValueError(f"duplicate prediction key: {reader}/{method}")
            if query_order is None:
                query_order = ids
            elif ids != query_order:
                raise ValueError(f"prediction order mismatch: {reader}/{method}")
            for row in cell:
                payload = {key: value for key, value in row.items() if key != "record_checksum"}
                if canonical_hash(payload) != row.get("record_checksum"):
                    raise ValueError(f"record checksum mismatch: {reader}/{method}/{row.get('query_id')}")
                key = (reader, method, str(row["query_id"]))
                if key in records:
                    raise ValueError(f"duplicate primary key: {key}")
                records[key] = row
    if len(records) != 2400:
        raise ValueError(f"expected 2400 predictions, found {len(records)}")
    return {"protocol_hash": protocol_hash, "query_order": query_order, "records": records}


def evaluate(args, sealed: dict) -> None:
    # The gold-bearing file is opened only below, after validate_all_cells returned 2400 sealed records.
    v20 = Path(__file__).resolve().parents[3]
    eval_dir = v20.parent / "v16_action_composition/evaluation"
    sys.path.insert(0, str(eval_dir))
    import joblib
    from eval_common import gold_support, normalize_title, official_metrics, source_documents, unit_features

    gold_rows = rows(args.gold_manifest)
    gold_by_id = {str(row.get("query_id", row.get("id", row.get("_id")))): row for row in gold_rows}
    if list(gold_by_id) != sealed["query_order"] or len(gold_by_id) != 300:
        raise ValueError("sealed gold manifest query order mismatch")
    support_checkpoint = joblib.load(eval_dir / "checkpoints/hotpotqa_support.joblib")
    contexts = {method: {str(row["query_id"]): row for row in rows(args.contexts_root / f"{method}.jsonl")} for method in METHODS}

    def predict_support(question: str, docs: list[dict], source_row: dict) -> set[tuple[str, int]]:
        local = {doc["doc_id"]: doc for doc in source_documents(source_row, "hotpotqa")}
        instances = []
        for doc_rank, doc in enumerate(docs):
            exact = local.get(doc["doc_id"])
            sentences = exact.get("sentences", []) if exact else [x.strip() for x in re.split(r"(?<=[.!?])\s+", str(doc["text"])) if x.strip()]
            sentences = sentences or [str(doc["text"])]
            for sent_id, sentence in enumerate(sentences):
                instances.append({
                    "identity": (normalize_title(doc["title"]), sent_id),
                    "features": unit_features(question, doc["title"], sentence, doc_rank, sent_id, len(sentences)),
                })
        probabilities = support_checkpoint["model"].predict_proba(np.asarray([row["features"] for row in instances]))[:, 1]
        ranked = sorted(zip(instances, probabilities), key=lambda pair: pair[1], reverse=True)
        selected = [row for row, score in ranked if score >= support_checkpoint["threshold"]][:6]
        minimum = int(support_checkpoint.get("minimum_predictions", 2))
        if len(selected) < minimum:
            selected = [row for row, _ in ranked[:minimum]]
        return {tuple(row["identity"]) for row in selected}

    per_query = []
    for reader in READERS:
        for method in METHODS:
            for query_id in sealed["query_order"]:
                prediction = sealed["records"][(reader, method, query_id)]["prediction"]
                gold_row = gold_by_id[query_id]
                context = contexts[method][query_id]
                support = predict_support(str(gold_row["question"]), context["reader_context_docs"], gold_row)
                scores = official_metrics(prediction, gold_row, support, "hotpotqa")
                per_query.append({"reader": reader, "method": method, "query_id": query_id, **scores})

    args.output_root.mkdir(parents=True, exist_ok=False)
    per_query_path = args.output_root / "reader_metrics_per_query.jsonl"
    with per_query_path.open("x", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    grouped = defaultdict(list)
    for row in per_query:
        grouped[(row["reader"], row["method"])].append(row)
    with (args.output_root / "reader_metrics.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reader", "method", "n", *METRICS])
        writer.writeheader()
        for (reader, method), values in sorted(grouped.items()):
            writer.writerow({"reader": reader, "method": method, "n": len(values), **{metric: np.mean([row[metric] for row in values]) for metric in METRICS}})

    rng = np.random.default_rng(20260812)
    bootstrap = []
    for reader in READERS:
        by_method = {method: {row["query_id"]: row for row in grouped[(reader, method)]} for method in METHODS}
        for left, right in COMPARISONS:
            for metric in METRICS:
                differences = np.asarray([by_method[left][qid][metric] - by_method[right][qid][metric] for qid in sealed["query_order"]], dtype=np.float64)
                sampled = differences[rng.integers(0, len(differences), size=(5000, len(differences)))].mean(axis=1)
                p_value = min(1.0, 2.0 * min(float(np.mean(sampled <= 0)), float(np.mean(sampled >= 0))))
                bootstrap.append({
                    "reader": reader, "comparison": f"{left}_minus_{right}", "metric": metric,
                    "mean_delta": float(differences.mean()), "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
                    "two_sided_p": p_value, "bootstrap_samples": 5000, "bootstrap_seed": 20260812, "effective_n": 300,
                })
    (args.output_root / "bootstrap_5000.json").write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n")
    checksums = {str(path.name): sha256(path) for path in args.output_root.iterdir() if path.is_file()}
    (args.output_root / "evaluation_checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--contexts-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sealed = validate_all_cells(args.predictions_root, args.contexts_root, args.protocol)
    evaluate(args, sealed)


if __name__ == "__main__":
    main()
