#!/usr/bin/env python3
"""Train fully nested missing-hop, semantic candidate, and pair-complementarity models."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from semantic_features import DOC_FEATURE_NAMES, PAIR_FEATURE_NAMES, QUERY_FEATURE_NAMES, build_query_cache, pair_key
from v4_common import (
    OUTPUTS, REPORTS, build_folds, context_from_snapshot, ensure_layout, grouped_outcomes,
    load_context_snapshots, load_source_examples, load_v3_merged_rows, query_fingerprint,
    sha256, write_json,
)


DEFAULT_BI_ENCODER = "/home/iiserver31/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130"
DEFAULT_CROSS_ENCODER = "/home/iiserver31/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/c5ee24cb16019beea0893ab7796b1df96625c6b8"
MISSING_TYPES = ["missing_bridge", "missing_answer_resolution", "redundant_context", "ordering_problem", "no_intervention_needed"]


def safe_pipeline(c_value: float, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=c_value, max_iter=3000, class_weight=class_weight, random_state=20260714)),
    ])


def inner_fold(query_id: str) -> int:
    return int(query_id[-6:], 16) % 3


def tune_binary_c(features: np.ndarray, labels: np.ndarray, groups: list[str]) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    if len(set(labels.tolist())) < 2:
        return 1.0, {"1.0": 0.0}
    for c_value in (0.1, 1.0, 10.0):
        predictions = np.zeros(len(labels), dtype=float)
        valid = np.zeros(len(labels), dtype=bool)
        for fold_id in range(3):
            train = np.array([inner_fold(group) != fold_id for group in groups])
            test = ~train
            if not test.any() or len(set(labels[train].tolist())) < 2:
                continue
            model = safe_pipeline(c_value)
            model.fit(features[train], labels[train])
            predictions[test] = model.predict_proba(features[test])[:, 1]
            valid[test] = True
        score = average_precision_score(labels[valid], predictions[valid]) if valid.any() and len(set(labels[valid].tolist())) > 1 else 0.0
        scores[str(c_value)] = float(score)
    best = max((float(key) for key in scores), key=lambda value: scores[str(value)])
    return best, scores


def tune_multiclass_c(features: np.ndarray, labels: np.ndarray, groups: list[str]) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for c_value in (0.1, 1.0, 10.0):
        fold_scores: list[float] = []
        for fold_id in range(3):
            train = np.array([inner_fold(group) != fold_id for group in groups])
            test = ~train
            if not test.any() or len(set(labels[train].tolist())) < 2:
                continue
            model = safe_pipeline(c_value)
            model.fit(features[train], labels[train])
            fold_scores.append(balanced_accuracy_score(labels[test], model.predict(features[test])))
        scores[str(c_value)] = float(np.mean(fold_scores)) if fold_scores else 0.0
    best = max((float(key) for key in scores), key=lambda value: scores[str(value)])
    return best, scores


def infer_missing_label(rows: list[dict[str, Any]]) -> str:
    baseline = next(row for row in rows if row["action_family"] == "fallback")
    actions = [row for row in rows if row["action_family"] != "fallback"]
    positive = [row for row in actions if bool(row.get("positive_action"))]
    if float(baseline["answer_f1"]) >= 1.0 - 1e-12 and float(baseline["title_recall"]) >= 1.0 - 1e-12:
        return "no_intervention_needed"
    if positive:
        best = max(positive, key=lambda row: (float(row.get("answer_title_product_delta", 0.0)), float(row.get("title_recall_delta", 0.0))))
        if best["action_family"] == "redundancy_aware_replacement":
            return "redundant_context"
        if best["action_family"] == "joint_reorder_and_insert" and float(best.get("title_recall_delta", 0.0)) <= 1e-12:
            return "ordering_problem"
        if float(best.get("title_recall_delta", 0.0)) > 1e-12 or len(best.get("added_doc_ids", [])) >= 2:
            return "missing_bridge"
        if float(best.get("answer_f1_delta", 0.0)) > 1e-12:
            return "missing_answer_resolution"
    if float(baseline["title_recall"]) < 1.0 - 1e-12:
        return "missing_bridge"
    if float(baseline["answer_f1"]) < 1.0 - 1e-12:
        return "missing_answer_resolution"
    return "no_intervention_needed"


def outcome_targets(grouped: dict[str, list[dict[str, Any]]], cache: dict[str, Any]) -> dict[str, Any]:
    missing_labels: dict[str, str] = {}
    doc_labels: dict[str, dict[str, int]] = {}
    pair_labels: dict[str, dict[str, int]] = {}
    for query_id, rows in grouped.items():
        missing_labels[query_id] = infer_missing_label(rows)
        candidate_ids = cache["queries"][query_id]["candidate_ids"]
        doc_labels[query_id] = {doc_id: 0 for doc_id in candidate_ids}
        pair_labels[query_id] = {}
        for row in rows:
            if row["action_family"] == "fallback":
                continue
            positive = int(bool(row.get("positive_action")))
            added = [doc_id for doc_id in row.get("added_doc_ids", []) if doc_id in doc_labels[query_id]]
            for doc_id in added:
                doc_labels[query_id][doc_id] = max(doc_labels[query_id][doc_id], positive)
            if len(added) == 2:
                key = pair_key(*added)
                pair_labels[query_id][key] = max(pair_labels[query_id].get(key, 0), positive)
    return {"missing": missing_labels, "doc": doc_labels, "pair": pair_labels}


def compute_cache(source: dict[str, Any], snapshots: dict[str, Any], bi_path: str, cross_path: str, device: str, batch_size: int) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    bi_encoder = SentenceTransformer(bi_path, device=device, local_files_only=True)
    cross_encoder = CrossEncoder(cross_path, device=device, local_files_only=True, max_length=512)
    query_ids = sorted(source)
    questions = [str(source[query_id]["question"]) for query_id in query_ids]
    query_embeddings = bi_encoder.encode(questions, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    cache_queries: dict[str, Any] = {}
    for start in range(0, len(query_ids), 64):
        batch_ids = query_ids[start:start + 64]
        all_docs: list[dict[str, Any]] = []
        doc_counts: list[int] = []
        cross_pairs: list[tuple[str, str]] = []
        batch_baselines: list[list[str]] = []
        for query_id in batch_ids:
            item = source[query_id]
            baseline = context_from_snapshot(snapshots[query_id], item["docs"])
            frozen = {doc["doc_id"]: doc for doc in baseline}
            docs = [frozen.get(doc["doc_id"], doc) for doc in item["docs"]]
            docs.extend(doc for doc in baseline if doc["doc_id"] not in {value["doc_id"] for value in item["docs"]})
            all_docs.extend(docs)
            doc_counts.append(len(docs))
            batch_baselines.append([str(doc["doc_id"]) for doc in baseline])
            cross_pairs.extend((str(item["question"]), f"{doc['title']}. {doc['text']}") for doc in docs)
        doc_texts = [f"{doc['title']}. {doc['text']}" for doc in all_docs]
        doc_embeddings = bi_encoder.encode(doc_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        cross_scores = cross_encoder.predict(cross_pairs, batch_size=batch_size, show_progress_bar=False)
        offset = 0
        for local_index, query_id in enumerate(batch_ids):
            count = doc_counts[local_index]
            query_index = start + local_index
            query_cache = build_query_cache(
                str(source[query_id]["question"]),
                all_docs[offset:offset + count],
                batch_baselines[local_index],
                np.asarray(query_embeddings[query_index]),
                np.asarray(doc_embeddings[offset:offset + count]),
                [float(value) for value in cross_scores[offset:offset + count]],
            )
            cache_queries[query_id] = query_cache
            offset += count
    return {
        "version": "v4_semantic_features_1",
        "bi_encoder_path": bi_path,
        "cross_encoder_path": cross_path,
        "doc_feature_names": DOC_FEATURE_NAMES,
        "query_feature_names": QUERY_FEATURE_NAMES,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "queries": cache_queries,
    }


def train_fold(fold: dict[str, Any], cache: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    train_ids = list(fold["train_query_ids"])
    query_x = np.asarray([cache["queries"][query_id]["query_features"] for query_id in train_ids], dtype=np.float32)
    query_y = np.asarray([targets["missing"][query_id] for query_id in train_ids])
    missing_c, missing_cv = tune_multiclass_c(query_x, query_y, train_ids)
    missing_model = safe_pipeline(missing_c)
    missing_model.fit(query_x, query_y)

    doc_x, doc_y, doc_groups = [], [], []
    pair_x, pair_y, pair_groups = [], [], []
    for query_id in train_ids:
        query_cache = cache["queries"][query_id]
        for doc_id in query_cache["candidate_ids"]:
            doc_x.append(query_cache["doc_features"][doc_id])
            doc_y.append(targets["doc"][query_id][doc_id])
            doc_groups.append(query_id)
        for key, label in targets["pair"][query_id].items():
            if key in query_cache["pair_features"]:
                pair_x.append(query_cache["pair_features"][key])
                pair_y.append(label)
                pair_groups.append(query_id)
    doc_x_array, doc_y_array = np.asarray(doc_x, dtype=np.float32), np.asarray(doc_y, dtype=int)
    doc_c, doc_cv = tune_binary_c(doc_x_array, doc_y_array, doc_groups)
    doc_model = safe_pipeline(doc_c)
    doc_model.fit(doc_x_array, doc_y_array)

    pair_x_array, pair_y_array = np.asarray(pair_x, dtype=np.float32), np.asarray(pair_y, dtype=int)
    if len(pair_y_array) and len(set(pair_y_array.tolist())) >= 2:
        pair_c, pair_cv = tune_binary_c(pair_x_array, pair_y_array, pair_groups)
        pair_model = safe_pipeline(pair_c)
        pair_model.fit(pair_x_array, pair_y_array)
    else:
        pair_c, pair_cv, pair_model = 1.0, {"1.0": 0.0}, None

    return {
        "missing_model": missing_model,
        "doc_model": doc_model,
        "pair_model": pair_model,
        "metadata": {
            "fold_id": fold["fold_id"],
            "train_query_fingerprint": query_fingerprint(train_ids),
            "test_query_fingerprint": query_fingerprint(fold["test_query_ids"]),
            "n_train_queries": len(train_ids),
            "missing_label_counts": dict(Counter(query_y.tolist())),
            "doc_training_rows": len(doc_y),
            "doc_positive_rows": int(doc_y_array.sum()),
            "pair_training_rows": len(pair_y),
            "pair_positive_rows": int(pair_y_array.sum()) if len(pair_y_array) else 0,
            "selected_c": {"missing": missing_c, "doc": doc_c, "pair": pair_c},
            "inner_cv": {"missing_balanced_accuracy": missing_cv, "doc_average_precision": doc_cv, "pair_average_precision": pair_cv},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bi-encoder", default=os.environ.get("V4_BI_ENCODER", DEFAULT_BI_ENCODER))
    parser.add_argument("--cross-encoder", default=os.environ.get("V4_CROSS_ENCODER", DEFAULT_CROSS_ENCODER))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    source, snapshots = load_source_examples(), load_context_snapshots()
    cache_path = OUTPUTS / "semantic_generator/semantic_feature_cache.joblib"
    if args.reuse_cache and cache_path.exists():
        cache = joblib.load(cache_path)
    else:
        cache = compute_cache(source, snapshots, args.bi_encoder, args.cross_encoder, args.device, args.batch_size)
        joblib.dump(cache, cache_path, compress=3)

    grouped = grouped_outcomes(load_v3_merged_rows())
    targets = outcome_targets(grouped, cache)
    folds = build_folds(source)
    fold_records = []
    for fold in folds:
        bundle = train_fold(fold, cache, targets)
        model_path = OUTPUTS / f"semantic_generator/fold_{fold['fold_id']}_generator.joblib"
        joblib.dump(bundle, model_path, compress=3)
        record = dict(bundle["metadata"])
        record.update({
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "test_query_ids": fold["test_query_ids"],
        })
        fold_records.append(record)

    manifest = {
        "status": "complete",
        "protocol": "five-fold fully nested generator",
        "outer_train_uses": ["v3 reader outcome diagnostics", "inference-safe semantic and lexical features"],
        "outer_test_uses": ["frozen models", "question", "document text", "baseline context", "retrieval and semantic scores"],
        "target_query_gold_used": False,
        "target_query_answer_used": False,
        "target_query_reader_outcome_used": False,
        "target_query_oracle_action_used": False,
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256(cache_path),
        "folds": fold_records,
    }
    write_json(OUTPUTS / "semantic_generator/foldwise_generator_models.json", manifest)
    write_json(OUTPUTS / "audits/generator_training_no_leak_audit.json", manifest)

    mean_doc_ap = np.mean([max(record["inner_cv"]["doc_average_precision"].values()) for record in fold_records])
    mean_pair_ap = np.mean([max(record["inner_cv"]["pair_average_precision"].values()) for record in fold_records])
    report = f"""# Semantic Generator Training Report

The generator uses frozen MPNet bi-encoder embeddings and a frozen MS MARCO cross-encoder, then learns three outer-fold-specific components from outer-train outcomes only: a missing-hop estimator, a semantic document opportunity model, and a pair-complementarity model.

- Outer folds: 5 x 800 train / 200 frozen test queries.
- Mean best inner-CV document average precision: **{mean_doc_ap:.4f}**.
- Mean best inner-CV pair average precision: **{mean_pair_ap:.4f}**.
- Target-query gold support, answer, reader outcomes, and oracle actions: **not used**.

Action generation remains pending until `04_generate_outer_fold_actions.py` applies each frozen fold model to its disjoint outer-test queries.
"""
    (REPORTS / "semantic_generator_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "folds": len(fold_records), "mean_doc_ap": mean_doc_ap, "mean_pair_ap": mean_pair_ap}, indent=2))


if __name__ == "__main__":
    main()
