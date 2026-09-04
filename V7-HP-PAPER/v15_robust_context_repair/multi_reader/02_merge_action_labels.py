#!/usr/bin/env python3
"""Merge reader labels with inference-safe action features for scorer training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fold(query_id):
    return int(hashlib.sha1(str(query_id).encode()).hexdigest(), 16) % 5


def tokens(value):
    return {token for token in re.findall(r"[A-Za-z0-9]+", str(value).lower()) if len(token) > 1}


def jaccard(left, right):
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def summarize(prefix, values, output):
    values = list(map(float, values))
    output[f"{prefix}_mean"] = sum(values) / len(values) if values else 0.0
    output[f"{prefix}_max"] = max(values, default=0.0)
    output[f"{prefix}_min"] = min(values, default=0.0)


def action_features(action, docs, question, baseline_ids):
    sequence = [docs[doc_id] for doc_id in action["doc_ids"]]
    added = [docs[doc_id] for doc_id in action.get("added_doc_ids", [])]
    removed = [docs[doc_id] for doc_id in action.get("removed_doc_ids", [])]
    features = {
        "cheap_score": float(action.get("cheap_score", 0.0)),
        "added_count": len(added),
        "removed_count": len(removed),
        "is_baseline": float(action.get("is_baseline", False)),
        "baseline_preservation": len(set(action["doc_ids"]) & set(baseline_ids)) / len(baseline_ids),
        "top2_anchor_preservation": len(set(action["doc_ids"][:2]) & set(baseline_ids[:2])) / 2.0,
        "baseline_order_agreement": sum(doc_id in baseline_ids and baseline_ids.index(doc_id) == index for index, doc_id in enumerate(action["doc_ids"])) / 5.0,
        "mean_query_doc_overlap": sum(jaccard(question, f"{doc['title']} {doc['text']}") for doc in sequence) / len(sequence),
        "max_query_title_overlap": max((jaccard(question, doc["title"]) for doc in sequence), default=0.0),
        "mean_pairwise_diversity": 0.0,
        "max_pairwise_title_bridge": 0.0,
        "position_weighted_hybrid": sum(float(doc.get("hybrid_score", 0.0)) / (index + 1) for index, doc in enumerate(sequence)),
        "position_weighted_cross": sum(float(doc.get("cross_score", 0.0)) / (index + 1) for index, doc in enumerate(sequence)),
        "mean_document_log_length": sum(math.log1p(len(tokens(doc.get("text", "")))) for doc in sequence) / len(sequence),
    }
    pair_similarity, title_bridge = [], []
    for left in range(len(sequence)):
        for right in range(left + 1, len(sequence)):
            pair_similarity.append(jaccard(f"{sequence[left]['title']} {sequence[left]['text']}", f"{sequence[right]['title']} {sequence[right]['text']}"))
            title_bridge.append(jaccard(sequence[left]["title"], sequence[right]["title"]))
    features["mean_pairwise_diversity"] = 1.0 - sum(pair_similarity) / len(pair_similarity) if pair_similarity else 0.0
    features["max_pairwise_title_bridge"] = max(title_bridge, default=0.0)
    for name in ("hybrid_score", "dense_score", "sparse_score", "cross_score"):
        summarize(f"sequence_{name}", [doc.get(name, 0.0) for doc in sequence], features)
        summarize(f"added_{name}", [doc.get(name, 0.0) for doc in added], features)
        summarize(f"removed_{name}", [doc.get(name, 0.0) for doc in removed], features)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reader", action="append", required=True, help="reader_name=path")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    actions = {(str(row["query_id"]), str(row["action_id"])): row for row in read_jsonl(args.actions)}
    pools = {str(row["query_id"]): {doc["doc_id"]: doc for doc in row["documents"]} for row in read_jsonl(args.pool)}
    questions = {str(row.get("query_id", row.get("_id", row.get("id")))): str(row["question"]) for row in read_jsonl(args.split)}
    baseline_ids = {}
    for (qid, _), action in actions.items():
        if action.get("is_baseline"):
            baseline_ids[qid] = list(action["doc_ids"])
    labels = {}
    for value in args.reader:
        name, path = value.split("=", 1)
        for row in read_jsonl(path):
            labels.setdefault((str(row["query_id"]), str(row["action_id"])), {})[name] = {key: row[key] for key in ("answer_delta", "sp_delta", "joint_delta", "answer_drop", "joint_drop")}
    rows = []
    readers = sorted(value.split("=", 1)[0] for value in args.reader)
    for key, per_reader in labels.items():
        if set(per_reader) != set(readers) or key not in actions:
            continue
        action = actions[key]
        features = action_features(action, pools[key[0]], questions[key[0]], baseline_ids[key[0]])
        rows.append({"query_id": key[0], "action_id": key[1], "features": features, "reader_labels": per_reader})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, selected in (("train", [row for row in rows if fold(row["query_id"]) != 0]), ("development", [row for row in rows if fold(row["query_id"]) == 0])):
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({"status": "complete", "rows": len(rows), "readers": readers, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
