#!/usr/bin/env python3
"""Apply the frozen five-fold semantic generator to the disjoint scale-up set."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from semantic_features import pair_key
from v4_common import OUTPUTS, build_folds, ensure_layout, query_fingerprint, read_json, read_jsonl, sha256, write_json, write_jsonl


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probability(model: Any, features: list[float]) -> float:
    values = model.predict_proba(np.asarray([features], dtype=np.float32))[0]
    classes = list(model.classes_)
    return float(values[classes.index(1)]) if 1 in classes else 0.0


def multiclass_probabilities(model: Any, features: list[float]) -> dict[str, float]:
    values = model.predict_proba(np.asarray([features], dtype=np.float32))[0]
    return {str(label): float(value) for label, value in zip(model.classes_, values)}


def source_and_snapshots(context_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    source, snapshots = {}, {}
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
                {"title": doc["title"], "sentences": [doc["text"]]} for doc in row["baseline_context"]
            ],
        }
    return source, snapshots


def generate_actions(query_id: str, fold_id: int, query: dict[str, Any], bundle: dict[str, Any], max_actions: int) -> list[dict[str, Any]]:
    docs_by_id = {str(doc["doc_id"]): doc for doc in query["docs"]}
    baseline_ids = list(query["baseline_ids"])
    candidate_ids = list(query["candidate_ids"])
    missing = multiclass_probabilities(bundle["missing_model"], query["query_features"])
    doc_scores = {doc_id: probability(bundle["doc_model"], query["doc_features"][doc_id]) for doc_id in candidate_ids}
    semantic_scores = {
        doc_id: 0.45 * query["doc_feature_details"][doc_id]["query_doc_cosine"]
        + 0.25 * query["doc_feature_details"][doc_id]["cross_encoder_relevance"]
        + 0.20 * query["doc_feature_details"][doc_id]["bridge_entity_match"]
        + 0.10 * query["doc_feature_details"][doc_id]["novel_information"]
        - 0.15 * query["doc_feature_details"][doc_id]["redundancy"]
        for doc_id in candidate_ids
    }
    ranked_candidates = sorted(candidate_ids, key=lambda doc_id: (0.65 * doc_scores[doc_id] + 0.35 * semantic_scores[doc_id]), reverse=True)
    pair_scores: dict[str, float] = {}
    for left_id, right_id in combinations(candidate_ids, 2):
        key = pair_key(left_id, right_id)
        learned = probability(bundle["pair_model"], query["pair_features"][key]) if bundle["pair_model"] is not None else 0.0
        pair_scores[key] = 0.65 * learned + 0.175 * doc_scores[left_id] + 0.175 * doc_scores[right_id]
    ranked_pairs = sorted(pair_scores, key=pair_scores.get, reverse=True)
    baseline_anchor = {
        doc_id: 0.55 * query["doc_feature_details"][doc_id]["anchor_proxy"]
        + 0.45 * query["doc_feature_details"][doc_id]["query_doc_cosine"]
        for doc_id in baseline_ids
    }
    removable_start = min(2, max(0, len(baseline_ids) - 1))
    removable_positions = sorted(range(removable_start, len(baseline_ids)), key=lambda index: baseline_anchor[baseline_ids[index]])
    if not removable_positions:
        removable_positions = [len(baseline_ids) - 1]
    weakest_tail = removable_positions[0]
    most_redundant = max(removable_positions, key=lambda index: query["doc_feature_details"][baseline_ids[index]]["redundancy"])
    pool: list[dict[str, Any]] = []
    seen = {tuple(baseline_ids)}

    def add(family: str, name: str, context_ids: list[str], added: list[str], removed: list[str], ordering: str, opportunity: float) -> None:
        key = tuple(context_ids)
        if key in seen or len(context_ids) != len(baseline_ids) or len(set(context_ids)) != len(context_ids):
            return
        seen.add(key)
        removal_risk = max((baseline_anchor[doc_id] for doc_id in removed), default=0.0)
        family_probability = {
            "single_complementary_insertion": missing.get("missing_bridge", 0.0),
            "anchor_preserving_replacement": missing.get("missing_answer_resolution", 0.0) + 0.5 * missing.get("missing_bridge", 0.0),
            "semantic_two_document_chain": missing.get("missing_bridge", 0.0),
            "redundancy_replacement": missing.get("redundant_context", 0.0),
            "bridge_first_reorder": missing.get("ordering_problem", 0.0) + 0.5 * missing.get("missing_bridge", 0.0),
            "answer_anchor_first_reorder": missing.get("ordering_problem", 0.0) + 0.5 * missing.get("missing_answer_resolution", 0.0),
        }[family]
        pool.append({
            "family": family,
            "name": name,
            "context_ids": context_ids,
            "added": added,
            "removed": removed,
            "ordering": ordering,
            "generator_score": float(0.55 * opportunity + 0.30 * family_probability - 0.15 * removal_risk),
            "removal_risk": float(removal_risk),
        })

    for rank, doc_id in enumerate(ranked_candidates[:3]):
        context = list(baseline_ids)
        removed = context[weakest_tail]
        context[weakest_tail] = doc_id
        add("single_complementary_insertion", f"semantic_single_{rank + 1}", context, [doc_id], [removed], "stable", doc_scores[doc_id])
    for rank, doc_id in enumerate(ranked_candidates[:3]):
        position = removable_positions[min(rank, len(removable_positions) - 1)]
        context = list(baseline_ids)
        removed = context[position]
        context[position] = doc_id
        add("anchor_preserving_replacement", f"anchor_preserve_{rank + 1}", context, [doc_id], [removed], "anchor_first", 0.7 * doc_scores[doc_id] + 0.3 * semantic_scores[doc_id])
    for rank, key in enumerate(ranked_pairs[:3]):
        left_id, right_id = key.split("|||")
        preserve = max(0, len(baseline_ids) - 2)
        add("semantic_two_document_chain", f"semantic_pair_{rank + 1}", baseline_ids[:preserve] + [left_id, right_id], [left_id, right_id], baseline_ids[preserve:], "anchor_first", pair_scores[key])
        if rank == 0 and len(baseline_ids) >= 5:
            add("semantic_two_document_chain", "semantic_pair_bridge_middle", baseline_ids[:2] + [left_id, right_id, baseline_ids[2]], [left_id, right_id], baseline_ids[3:], "bridge_middle", pair_scores[key] - 0.02)
    if ranked_candidates:
        doc_id = ranked_candidates[0]
        context = list(baseline_ids)
        removed = context[most_redundant]
        context[most_redundant] = doc_id
        add("redundancy_replacement", "replace_semantic_redundancy", context, [doc_id], [removed], "stable", doc_scores[doc_id] + 0.15 * query["doc_feature_details"][removed]["redundancy"])
        if len(baseline_ids) >= 4:
            bridge_id = max(ranked_candidates, key=lambda value: query["doc_feature_details"][value]["bridge_entity_match"])
            add("bridge_first_reorder", "learned_bridge_first", [baseline_ids[0], bridge_id] + baseline_ids[1:-1], [bridge_id], [baseline_ids[-1]], "bridge_first", doc_scores[bridge_id])
            anchor_prefix = baseline_ids[: min(3, len(baseline_ids) - 1)]
            anchor_order = sorted(anchor_prefix, key=baseline_anchor.get, reverse=True)
            add("answer_anchor_first_reorder", "learned_anchor_order", anchor_order + [doc_id] + baseline_ids[len(anchor_prefix):-1], [doc_id], [baseline_ids[-1]], "answer_anchor_first", doc_scores[doc_id])

    no_intervention = missing.get("no_intervention_needed", 0.0)
    action_budget = min(max_actions, 4 if no_intervention >= 0.65 else 6 if no_intervention >= 0.45 else max_actions)
    selected = sorted(pool, key=lambda row: row["generator_score"], reverse=True)[:action_budget]
    rows = []
    for action_index, candidate in enumerate(selected):
        context_ids = candidate["context_ids"]
        rows.append({
            "query_id": query_id,
            "question": query["question"],
            "action_id": f"{query_id}::v4scale::{action_index:02d}",
            "outer_fold": fold_id,
            "action_family": candidate["family"],
            "action_name": candidate["name"],
            "context_doc_ids": context_ids,
            "context_titles": [docs_by_id[doc_id]["title"] for doc_id in context_ids],
            "context_docs": [docs_by_id[doc_id] for doc_id in context_ids],
            "added_doc_ids": candidate["added"],
            "removed_doc_ids": candidate["removed"],
            "ordering": candidate["ordering"],
            "generator_score": candidate["generator_score"],
            "is_new_vs_v3_action_table": True,
            "inference_safe_features": {
                "missing_hop_probabilities": missing,
                "added_doc_opportunity_mean": float(np.mean([doc_scores[doc_id] for doc_id in candidate["added"]])) if candidate["added"] else 0.0,
                "added_doc_semantic_mean": float(np.mean([semantic_scores[doc_id] for doc_id in candidate["added"]])) if candidate["added"] else 0.0,
                "removal_risk": candidate["removal_risk"],
            },
        })
    rows.append({
        "query_id": query_id,
        "question": query["question"],
        "action_id": f"{query_id}::v4scale::fallback",
        "outer_fold": fold_id,
        "action_family": "fallback",
        "action_name": "baseline_fallback",
        "context_doc_ids": baseline_ids,
        "context_titles": [docs_by_id[doc_id]["title"] for doc_id in baseline_ids],
        "context_docs": [docs_by_id[doc_id] for doc_id in baseline_ids],
        "added_doc_ids": [],
        "removed_doc_ids": [],
        "ordering": "stable",
        "generator_score": 0.0,
        "is_new_vs_v3_action_table": False,
        "inference_safe_features": {"missing_hop_probabilities": missing, "removal_risk": 0.0},
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    scale_dir = OUTPUTS / "scaleup"
    context_path = scale_dir / "frozen_baseline_contexts_3000.jsonl"
    context_audit = read_json(scale_dir / "same_source_context_audit.json")
    if context_audit.get("status") != "pass" or context_audit.get("baseline_1000_reproduction_rate") != 1.0:
        raise AssertionError("Exact 1,000-query baseline reproduction audit must pass before generation")
    context_rows = read_jsonl(context_path)
    source, snapshots = source_and_snapshots(context_rows)
    cache_path = scale_dir / "semantic_feature_cache_3000.joblib"
    train_module = load_module(Path(__file__).with_name("03_train_semantic_candidate_generator.py"), "v4_scale_train")
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

    manifest = read_json(OUTPUTS / "semantic_generator/foldwise_generator_models.json")
    folds = build_folds(source)
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    fold_records = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        model_record = next(row for row in manifest["folds"] if int(row["fold_id"]) == fold_id)
        bundle = joblib.load(model_record["model_path"])
        for query_id in fold["test_query_ids"]:
            generated = generate_actions(query_id, fold_id, cache["queries"][query_id], bundle, args.max_actions)
            rows.extend(generated)
            family_counts.update(row["action_family"] for row in generated if row["action_family"] != "fallback")
        fold_records.append({
            "fold_id": fold_id,
            "n_scale_queries": len(fold["test_query_ids"]),
            "scale_query_fingerprint": query_fingerprint(fold["test_query_ids"]),
            "frozen_model_path": model_record["model_path"],
            "frozen_model_sha256": model_record["model_sha256"],
            "model_retrained_on_scaleup": False,
        })
    output_path = scale_dir / "generated_actions_3000.jsonl"
    write_jsonl(output_path, rows)
    effective = [row for row in rows if row["action_family"] != "fallback"]
    audit = {
        "status": "pass",
        "protocol": "frozen five-fold generator deployment on disjoint same-source queries",
        "n_queries": len(source),
        "n_action_rows": len(rows),
        "n_effective_actions": len(effective),
        "max_actions_per_query": args.max_actions,
        "family_counts": dict(family_counts),
        "generator_or_encoders_retuned": False,
        "scaleup_gold_or_answer_used": False,
        "scaleup_reader_outcome_used": False,
        "v3_membership_feature_policy": "true for effective actions because all scale queries are disjoint from the v3 table",
        "feature_cache_path": str(cache_path),
        "feature_cache_sha256": sha256(cache_path),
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "folds": fold_records,
    }
    write_json(scale_dir / "frozen_generator_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
