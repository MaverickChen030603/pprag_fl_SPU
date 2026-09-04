#!/usr/bin/env python3
"""Deploy the unchanged HotpotQA V4 generator and selector on 2Wiki."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

from completion_common import (
    EXTERNAL,
    V4_ROOT,
    add_v4_import_path,
    ensure_layout,
    load_module,
    query_fingerprint,
    read_json,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
)


def source_and_snapshots(context_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    source: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    for row in context_rows:
        query_id = str(row["query_id"])
        source[query_id] = {
            "query_id": query_id,
            "question": str(row["question"]),
            "docs": list(row["all_docs"]),
        }
        snapshots[query_id] = {
            "query_id": query_id,
            "baseline_titles": list(row["baseline_titles"]),
            "baseline_context": [
                {"title": doc["title"], "sentences": list(doc.get("sentences", [doc["text"]]))}
                for doc in row["baseline_context"]
            ],
        }
    return source, snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    add_v4_import_path()
    from v4_common import build_folds

    data_audit = read_json(EXTERNAL / "data_and_baseline_audit.json")
    if data_audit.get("status") != "pass" or data_audit.get("labels_used_for_retrieval"):
        raise AssertionError("2Wiki data/baseline no-leak audit must pass")
    contexts = read_jsonl(EXTERNAL / "frozen_baseline_contexts_1000.jsonl")
    source, snapshots = source_and_snapshots(contexts)
    cache_path = EXTERNAL / "semantic_feature_cache_1000.joblib"
    train_module = load_module(V4_ROOT / "03_train_semantic_candidate_generator.py", "v4_external_train")
    if args.reuse_cache and cache_path.exists():
        cache = joblib.load(cache_path)
    else:
        cache = train_module.compute_cache(
            source,
            snapshots,
            train_module.DEFAULT_BI_ENCODER,
            train_module.DEFAULT_CROSS_ENCODER,
            args.device,
            args.batch_size,
        )
        joblib.dump(cache, cache_path, compress=3)

    generate_module = load_module(V4_ROOT / "14_generate_frozen_scaleup_actions.py", "v4_external_generate")
    manifest = read_json(V4_ROOT / "outputs/semantic_generator/foldwise_generator_models.json")
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    fold_records = []
    for fold in build_folds(source):
        fold_id = int(fold["fold_id"])
        record = next(value for value in manifest["folds"] if int(value["fold_id"]) == fold_id)
        bundle = joblib.load(record["model_path"])
        for query_id in fold["test_query_ids"]:
            generated = generate_module.generate_actions(query_id, fold_id, cache["queries"][query_id], bundle, 8)
            for row in generated:
                row["action_id"] = str(row["action_id"]).replace("::v4scale::", "::v4xfer::")
                row["is_new_vs_v3_action_table"] = True
            rows.extend(generated)
            family_counts.update(row["action_family"] for row in generated if row["action_family"] != "fallback")
        fold_records.append({
            "fold_id": fold_id,
            "n_target_queries": len(fold["test_query_ids"]),
            "target_query_fingerprint": query_fingerprint(fold["test_query_ids"]),
            "hotpot_model_sha256": record["model_sha256"],
            "target_labels_used": False,
            "target_outcomes_used": False,
        })
    actions_path = EXTERNAL / "generated_actions_1000.jsonl"
    write_jsonl(actions_path, rows)

    selector_module = load_module(V4_ROOT / "07_train_nested_selector_v4.py", "v4_external_selector")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baselines: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if row["action_family"] == "fallback":
            baselines[query_id] = row
        else:
            grouped[query_id].append(row)

    selected_by_query: dict[str, dict[str, Any]] = {}
    selector_records = []
    for fold in build_folds(source):
        fold_id = int(fold["fold_id"])
        bundle_path = V4_ROOT / f"outputs/scaleup/selector_models/fold_{fold_id}_selector.joblib"
        selector_bundle = joblib.load(bundle_path)
        query_ids = set(fold["test_query_ids"])
        test_rows = [row for query_id in query_ids for row in grouped[query_id]]
        safe = selector_module.probabilities(selector_bundle["safety_model"], test_rows)
        positive = selector_module.probabilities(selector_bundle["opportunity_model"], test_rows)
        scored = []
        for row, safe_prob, positive_prob in zip(test_rows, safe, positive):
            value = dict(row)
            value["pred_answer_safe_prob"] = float(safe_prob)
            value["pred_positive_prob"] = float(positive_prob)
            scored.append(value)
        config = selector_bundle["config"]
        selected = selector_module.select(
            scored,
            query_ids,
            float(config["safe_threshold"]),
            float(config["positive_threshold"]),
            float(config["coverage"]),
        )
        selected_by_query.update(selected)
        selector_records.append({
            "fold_id": fold_id,
            "model_sha256": sha256(bundle_path),
            "frozen_config": config,
            "n_target_queries": len(query_ids),
            "n_selected": len(selected),
            "target_labels_or_outcomes_used": False,
        })

    selections = []
    for query_id in sorted(baselines):
        baseline = baselines[query_id]
        selected = selected_by_query.get(query_id)
        current = selected or baseline
        selections.append({
            "query_id": query_id,
            "outer_fold": int(current["outer_fold"]),
            "selected": selected is not None,
            "fallback": selected is None,
            "action_id": str(current["action_id"]),
            "action_family": str(current["action_family"]),
            "context_doc_ids": list(current["context_doc_ids"]),
            "context_titles": list(current["context_titles"]),
            "pred_answer_safe_prob": float(current.get("pred_answer_safe_prob", 0.0)),
            "pred_positive_prob": float(current.get("pred_positive_prob", 0.0)),
        })
    selections_path = EXTERNAL / "frozen_selector_selections_1000.jsonl"
    write_jsonl(selections_path, selections)
    selected_count = sum(bool(row["selected"]) for row in selections)
    audit = {
        "status": "pass",
        "protocol": "HotpotQA-frozen V4 zero-shot transfer to 2WikiMultiHopQA",
        "n_queries": len(source),
        "n_action_rows": len(rows),
        "n_effective_actions": sum(row["action_family"] != "fallback" for row in rows),
        "family_counts": dict(family_counts),
        "selected_count": selected_count,
        "coverage": selected_count / len(source),
        "semantic_generator_frozen": True,
        "missing_hop_estimator_frozen": True,
        "document_opportunity_model_frozen": True,
        "pair_model_frozen": True,
        "selector_models_frozen": True,
        "thresholds_and_coverage_frozen": True,
        "target_labels_used_for_features_generation_or_selection": False,
        "target_reader_outcomes_used_for_generation_or_selection": False,
        "target_tuning": False,
        "feature_cache_sha256": sha256(cache_path),
        "actions_sha256": sha256(actions_path),
        "selections_sha256": sha256(selections_path),
        "generator_folds": fold_records,
        "selector_folds": selector_records,
    }
    write_json(EXTERNAL / "frozen_generator_selector_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
