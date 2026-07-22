#!/usr/bin/env python3
"""Train a frozen sentence/paragraph support predictor for reader evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_common import gold_support, normalize_title, source_documents, support_metrics, unit_features


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def units(row, dataset):
    gold = gold_support(row, dataset)
    output = []
    for doc_rank, doc in enumerate(source_documents(row, dataset)):
        if dataset == "musique":
            values = [(doc["paragraph_idx"], doc["text"])]
        else:
            values = list(enumerate(doc.get("sentences", [])))
        for unit_rank, text in values:
            identity = ("paragraph", int(unit_rank)) if dataset == "musique" else (normalize_title(doc["title"]), int(unit_rank))
            output.append({"query_id": str(row["query_id"]), "identity": identity, "features": unit_features(str(row["question"]), doc["title"], str(text), doc_rank, unit_rank, len(values)), "label": int(identity in gold)})
    return output


def predicted_set(model, values, threshold, minimum):
    probability = model.predict_proba(np.asarray([row["features"] for row in values]))[:, 1]
    ranked = sorted(zip(values, probability), key=lambda pair: pair[1], reverse=True)
    selected = [row for row, score in ranked if score >= threshold][:6]
    if len(selected) < minimum:
        selected = [row for row, _ in ranked[:minimum]]
    return {tuple(row["identity"]) for row in selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=5000)
    args = parser.parse_args()
    train_rows, dev_rows = read(args.train)[:args.max_train], read(args.development)
    train_units = [unit for row in train_rows for unit in units(row, args.dataset)]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=20260722))
    model.fit(np.asarray([row["features"] for row in train_units]), np.asarray([row["label"] for row in train_units]))
    dev_by_id = {str(row["query_id"]): row for row in dev_rows}
    grouped = defaultdict(list)
    for row in dev_rows:
        grouped[str(row["query_id"])].extend(units(row, args.dataset))
    minimum = 2
    candidates = []
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        scores = [support_metrics(predicted_set(model, values, threshold, minimum), gold_support(dev_by_id[qid], args.dataset))["f1"] for qid, values in grouped.items()]
        candidates.append((float(np.mean(scores)), threshold))
    dev_f1, threshold = max(candidates)
    payload = {"model": model, "threshold": threshold, "dataset": args.dataset, "unit": "paragraph" if args.dataset == "musique" else "sentence", "feature_dim": 9, "minimum_predictions": minimum, "train_queries": len(train_rows), "development_queries": len(dev_rows), "development_sp_f1": dev_f1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output)
    print(json.dumps({key: value for key, value in payload.items() if key != "model"}, indent=2))


if __name__ == "__main__":
    main()
