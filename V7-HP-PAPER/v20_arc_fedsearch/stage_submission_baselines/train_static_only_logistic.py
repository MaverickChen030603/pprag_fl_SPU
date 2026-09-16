#!/usr/bin/env python3
"""Train B2 with only the coordinator-side static score."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True
    )
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--v16-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sys.path.insert(0, str(args.v16_eval))
    from eval_common import document_id

    def support_documents(row: dict[str, Any]) -> set[str]:
        if args.dataset == "musique":
            return {
                document_id(
                    args.dataset,
                    str(paragraph.get("title", "")),
                    str(paragraph.get("paragraph_text", "")),
                )
                for paragraph in row.get("paragraphs", [])
                if paragraph.get("is_supporting", paragraph.get("is_support", False))
            }
        facts = row.get("supporting_facts", {})
        titles = (
            facts.get("title", [])
            if isinstance(facts, dict)
            else [value[0] for value in facts if value]
        )
        return {document_id(args.dataset, str(title)) for title in titles}

    assignment = {
        str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment)
    }
    split_rows = {query_id(row): row for row in rows(args.split)}
    packet_rows = list(rows(args.packets))
    if set(split_rows) != {str(row["query_id"]) for row in packet_rows}:
        raise ValueError("training split and packet query IDs differ")

    examples = []
    labels = []
    positive_queries = 0
    for packet in packet_rows:
        qid = str(packet["query_id"])
        support_clients = {
            assignment[document]
            for document in support_documents(split_rows[qid])
            if document in assignment
        }
        positive_queries += bool(support_clients)
        for record in packet["p0_candidate_records"]:
            examples.append([float(record["static_score"])])
            labels.append(int(int(record["client_id"]) in support_clients))
    matrix = np.asarray(examples, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if set(target.tolist()) != {0, 1}:
        raise ValueError("training rows must contain both classes")

    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=args.seed,
    )
    model.fit(scaler.transform(matrix), target)
    probabilities = model.predict_proba(scaler.transform(matrix))[:, 1]
    payload = {
        "model": model,
        "scaler": scaler,
        "feature_names": ["static_score"],
        "seed": args.seed,
        "dataset": args.dataset,
        "information_boundary": "coordinator_static_score_only",
    }
    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / "static_only_logistic.pkl"
    with model_path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "status": "trained_not_evaluated",
        "dataset": args.dataset,
        "model": "static_only_logistic",
        "seed": args.seed,
        "queries": len(split_rows),
        "queries_with_mapped_support_client": positive_queries,
        "candidate_rows": len(target),
        "positive_rows": int(target.sum()),
        "feature_count": 1,
        "feature_names": ["static_score"],
        "train_auc": float(roc_auc_score(target, probabilities)),
        "train_auprc": float(average_precision_score(target, probabilities)),
        "split_sha256": sha256(args.split),
        "packets_sha256": sha256(args.packets),
        "assignment_sha256": sha256(args.assignment),
        "model_sha256": sha256(model_path),
        "test_or_confirmation_labels_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
