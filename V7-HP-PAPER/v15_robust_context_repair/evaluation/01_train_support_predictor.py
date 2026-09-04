#!/usr/bin/env python3
"""Train and freeze an inference-safe sentence support predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retrieval"))
from retrieval_common import documents, iter_rows, query_id
from eval_common import gold_support, normalize_title, sentence_features, support_metrics


def instances(rows, dataset):
    output = []
    for row in rows:
        gold = gold_support(row)
        for rank, doc in enumerate(documents(row, dataset)):
            sentences = doc.get("sentences", [])
            for sent_id, sentence in enumerate(sentences):
                output.append({"query_id": query_id(row), "title": doc["title"], "sent_id": sent_id, "features": sentence_features(str(row["question"]), doc["title"], str(sentence), rank, sent_id, len(sentences)), "label": int((normalize_title(doc["title"]), sent_id) in gold)})
    return output


def predict_set(model, values, threshold):
    probability = model.predict_proba(np.asarray([row["features"] for row in values]))[:, 1]
    ranked = sorted(zip(values, probability), key=lambda pair: pair[1], reverse=True)
    chosen = [row for row, score in ranked if score >= threshold][:5]
    if len(chosen) < 2:
        chosen = [row for row, _ in ranked[:2]]
    return {(normalize_title(row["title"]), int(row["sent_id"])) for row in chosen}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=5000)
    args = parser.parse_args()
    train_rows = list(iter_rows(args.train))[:args.max_train]
    dev_rows = list(iter_rows(args.development))
    train_instances = instances(train_rows, args.dataset)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=20260721))
    model.fit(np.asarray([row["features"] for row in train_instances]), np.asarray([row["label"] for row in train_instances]))
    dev_by_id = {query_id(row): row for row in dev_rows}
    dev_instances = instances(dev_rows, args.dataset)
    grouped = {}
    for row in dev_instances:
        grouped.setdefault(row["query_id"], []).append(row)
    candidates = []
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        scores = [support_metrics(predict_set(model, values, threshold), gold_support(dev_by_id[qid]))["f1"] for qid, values in grouped.items()]
        candidates.append((float(np.mean(scores)), threshold))
    dev_f1, threshold = max(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "dataset": args.dataset, "feature_dim": 9, "train_queries": len(train_rows), "development_queries": len(dev_rows), "development_sp_f1": dev_f1}, args.output)
    print(json.dumps({"status": "complete", "threshold": threshold, "development_sp_f1": dev_f1, "train_instances": len(train_instances), "checkpoint": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

