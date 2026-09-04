#!/usr/bin/env python3
"""Train a query-level cheap opportunity gate from reader-labelled actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALLOWED_EXACT = {
    "mean_query_doc_overlap",
    "max_query_title_overlap",
    "mean_pairwise_diversity",
    "max_pairwise_title_bridge",
    "mean_document_log_length",
    "position_weighted_hybrid",
}
ALLOWED_PREFIXES = ("sequence_hybrid_", "sequence_dense_", "sequence_sparse_")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cheap_names(rows: list[dict]) -> list[str]:
    names = set()
    for row in rows:
        names.update(row.get("features", {}))
    selected = sorted(name for name in names if name in ALLOWED_EXACT or name.startswith(ALLOWED_PREFIXES))
    blocked = [name for name in selected if "cross" in name.lower() or "reader" in name.lower()]
    if blocked:
        raise ValueError(f"expensive/outcome features reached cheap gate: {blocked}")
    return selected


def build_examples(rows: list[dict], feature_names: list[str], beta: float, min_gain: float):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["query_id"])].append(row)
    x, y, query_ids = [], [], []
    for query_id, actions in groups.items():
        baseline = next(row for row in actions if float(row["features"].get("is_baseline", 0.0)) > 0.5)
        readers = sorted(baseline["reader_labels"])
        best = -float("inf")
        for action in actions:
            deltas = np.asarray([float(action["reader_labels"][reader]["joint_delta"]) for reader in readers])
            best = max(best, float(deltas.mean() - beta * deltas.std()))
        x.append([float(baseline["features"].get(name, 0.0)) for name in feature_names])
        y.append(int(best > min_gain))
        query_ids.append(query_id)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), query_ids


def choose_threshold(y: np.ndarray, probability: np.ndarray, target_recall: float) -> tuple[float, dict]:
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 201):
        selected = probability >= threshold
        positives = y == 1
        recall = float((selected & positives).sum() / max(1, positives.sum()))
        invocation = float(selected.mean())
        candidates.append((recall >= target_recall, -invocation, float(threshold), recall, invocation))
    feasible = [item for item in candidates if item[0]]
    chosen = max(feasible or candidates, key=lambda item: (item[0], item[1], item[2]))
    return chosen[2], {"opportunity_recall": chosen[3], "invocation_rate": chosen[4]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--target-recall", type=float, default=0.90)
    args = parser.parse_args()

    train_rows, dev_rows = read_jsonl(args.train), read_jsonl(args.development)
    names = cheap_names(train_rows + dev_rows)
    train_x, train_y, _ = build_examples(train_rows, names, args.beta, args.min_gain)
    dev_x, dev_y, dev_ids = build_examples(dev_rows, names, args.beta, args.min_gain)
    if len(np.unique(train_y)) < 2:
        raise ValueError("cheap gate requires both opportunity and no-opportunity training queries")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260721)),
        ]
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(dev_x)[:, 1]
    threshold, dev_metrics = choose_threshold(dev_y, probability, args.target_recall)
    artifact = {
        "model": model,
        "feature_names": names,
        "threshold": threshold,
        "beta": args.beta,
        "min_gain": args.min_gain,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    manifest = {
        "status": "complete",
        "train_queries": int(len(train_y)),
        "development_queries": int(len(dev_y)),
        "development_query_ids_sha256": hashlib.sha256("\n".join(sorted(dev_ids)).encode("utf-8")).hexdigest(),
        "feature_names": names,
        "threshold": threshold,
        "target_recall": args.target_recall,
        "development": dev_metrics,
        "positive_rate_train": float(train_y.mean()),
        "positive_rate_development": float(dev_y.mean()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
