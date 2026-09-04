#!/usr/bin/env python3
"""Build fully nested Lite pair-complementary action generators on Hotpot dev."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v5_common import HERE, V4, V4_COMPLETION, config, read_json, read_jsonl, write_json, write_jsonl


OUT = HERE / "outputs" / "lite_model"
VARIANTS = ("lite_lexical_pair", "lite_semantic_pair", "pairchain_ablation")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_project_module(path: Path, name: str, project_root: Path) -> Any:
    """Load legacy modules whose helpers still resolve assets from Path.cwd()."""
    previous_cwd = Path.cwd()
    try:
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        os.chdir(project_root)
        return load_module(path, name)
    finally:
        os.chdir(previous_cwd)


def pair_key(left: str, right: str) -> str:
    return "|||".join(sorted((str(left), str(right))))


def lexical_doc_score(details: dict[str, float]) -> float:
    return float(
        0.34 * details["bm25"]
        + 0.18 * details["query_overlap"]
        + 0.12 * details["title_overlap"]
        + 0.13 * details["entity_overlap"]
        + 0.15 * details["bridge_entity_match"]
        + 0.08 * details["novel_information"]
        - 0.10 * details["redundancy"]
    )


def pair_features(query: dict[str, Any], left_id: str, right_id: str, semantic: bool) -> list[float]:
    left = query["doc_feature_details"][left_id]
    right = query["doc_feature_details"][right_id]
    lexical_left, lexical_right = lexical_doc_score(left), lexical_doc_score(right)
    values = [
        lexical_left,
        lexical_right,
        lexical_left + lexical_right,
        min(lexical_left, lexical_right),
        left["bm25"] + right["bm25"],
        left["query_overlap"] + right["query_overlap"],
        left["title_overlap"] + right["title_overlap"],
        left["entity_overlap"] + right["entity_overlap"],
        left["bridge_entity_match"] + right["bridge_entity_match"],
        left["novel_information"] + right["novel_information"],
        left["redundancy"] + right["redundancy"],
        query["pair_features"][pair_key(left_id, right_id)][8],
    ]
    if semantic:
        values.extend(
            [
                left["query_doc_cosine"],
                right["query_doc_cosine"],
                left["query_doc_cosine"] + right["query_doc_cosine"],
                min(left["query_doc_cosine"], right["query_doc_cosine"]),
                query["pair_features"][pair_key(left_id, right_id)][6],
            ]
        )
    return [float(value) for value in values]


def fit_pair_model(
    cache: dict[str, Any], pair_targets: dict[str, dict[str, int]], train_ids: list[str], semantic: bool
) -> Any:
    x, y = [], []
    for query_id in train_ids:
        query = cache["queries"][query_id]
        for left_id, right_id in combinations(query["candidate_ids"], 2):
            key = pair_key(left_id, right_id)
            x.append(pair_features(query, left_id, right_id, semantic))
            y.append(int(pair_targets.get(query_id, {}).get(key, 0)))
    array = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=int)
    if not len(labels):
        raise AssertionError("No pair training rows")
    if len(set(labels.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(labels[0]))
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=20260714),
        )
    model.fit(array, labels)
    return model


def probability(model: Any, features: list[float]) -> float:
    values = model.predict_proba(np.asarray([features], dtype=np.float64))[0]
    classes = list(model.classes_)
    return float(values[classes.index(1)]) if 1 in classes else 0.0


def generate_actions(
    query_id: str,
    fold_id: int,
    query: dict[str, Any],
    model: Any,
    variant: str,
    max_actions: int,
) -> list[dict[str, Any]]:
    semantic = variant == "lite_semantic_pair"
    pair_only = variant == "pairchain_ablation"
    docs = {str(row["doc_id"]): row for row in query["docs"]}
    baseline = list(query["baseline_ids"])
    candidates = list(query["candidate_ids"])
    if not baseline:
        raise AssertionError(query_id)
    doc_scores = {}
    for doc_id in candidates:
        details = query["doc_feature_details"][doc_id]
        score = lexical_doc_score(details)
        if semantic:
            score = 0.72 * score + 0.28 * float(details["query_doc_cosine"])
        doc_scores[doc_id] = float(score)
    ranked_docs = sorted(candidates, key=lambda doc_id: (doc_scores[doc_id], doc_id), reverse=True)
    pair_scores: dict[str, float] = {}
    for left_id, right_id in combinations(candidates, 2):
        key = pair_key(left_id, right_id)
        learned = probability(model, pair_features(query, left_id, right_id, semantic))
        pair_scores[key] = float(0.70 * learned + 0.15 * doc_scores[left_id] + 0.15 * doc_scores[right_id])
    ranked_pairs = sorted(pair_scores, key=lambda key: (pair_scores[key], key), reverse=True)
    baseline_anchor = {
        doc_id: float(query["doc_feature_details"][doc_id]["anchor_proxy"])
        + (0.12 * float(query["doc_feature_details"][doc_id]["query_doc_cosine"]) if semantic else 0.0)
        for doc_id in baseline
    }
    lock = min(2, max(0, len(baseline) - 1))
    removable = sorted(range(lock, len(baseline)), key=lambda index: baseline_anchor[baseline[index]])
    if not removable:
        removable = [len(baseline) - 1]
    pool: list[dict[str, Any]] = []
    seen = {tuple(baseline)}

    def add(family: str, name: str, context_ids: list[str], added: list[str], removed: list[str], score: float) -> None:
        signature = tuple(context_ids)
        if signature in seen or len(context_ids) != len(baseline) or len(set(context_ids)) != len(context_ids):
            return
        seen.add(signature)
        removal_risk = max((baseline_anchor[doc_id] for doc_id in removed), default=0.0)
        pool.append(
            {
                "family": family,
                "name": name,
                "context_ids": context_ids,
                "added": added,
                "removed": removed,
                "score": float(score - 0.12 * removal_risk),
                "removal_risk": float(removal_risk),
            }
        )

    if not pair_only:
        for rank, doc_id in enumerate(ranked_docs[:2]):
            position = removable[min(rank, len(removable) - 1)]
            context_ids = list(baseline)
            removed = context_ids[position]
            context_ids[position] = doc_id
            add(
                "anchor_preserving_replacement",
                f"{variant}_anchor_{rank + 1}",
                context_ids,
                [doc_id],
                [removed],
                doc_scores[doc_id],
            )
    for rank, key in enumerate(ranked_pairs[:4]):
        left_id, right_id = key.split("|||", 1)
        preserve = max(0, len(baseline) - 2)
        add(
            "semantic_two_document_chain",
            f"{variant}_pair_{rank + 1}",
            baseline[:preserve] + [left_id, right_id],
            [left_id, right_id],
            baseline[preserve:],
            pair_scores[key],
        )
        if rank == 0 and len(baseline) >= 5:
            add(
                "semantic_two_document_chain",
                f"{variant}_pair_bridge_middle",
                baseline[:2] + [left_id, right_id, baseline[2]],
                [left_id, right_id],
                baseline[3:],
                pair_scores[key] - 0.02,
            )
    selected = sorted(pool, key=lambda row: row["score"], reverse=True)[:max_actions]
    rows = []
    for index, candidate in enumerate(selected):
        added = candidate["added"]
        rows.append(
            {
                "query_id": query_id,
                "question": query["question"],
                "action_id": f"{query_id}::v5lite::{variant}::{index:02d}",
                "outer_fold": fold_id,
                "variant": variant,
                "action_family": candidate["family"],
                "action_name": candidate["name"],
                "context_doc_ids": candidate["context_ids"],
                "context_titles": [docs[doc_id]["title"] for doc_id in candidate["context_ids"]],
                "context_docs": [docs[doc_id] for doc_id in candidate["context_ids"]],
                "added_doc_ids": added,
                "removed_doc_ids": candidate["removed"],
                "ordering": "anchor_first",
                "generator_score": candidate["score"],
                "is_new_vs_v3_action_table": True,
                "inference_safe_features": {
                    "missing_hop_probabilities": {},
                    "added_doc_opportunity_mean": float(np.mean([doc_scores[doc_id] for doc_id in added])) if added else 0.0,
                    "added_doc_semantic_mean": float(np.mean([query["doc_feature_details"][doc_id]["query_doc_cosine"] for doc_id in added])) if semantic and added else 0.0,
                    "removal_risk": candidate["removal_risk"],
                    "pair_complementarity": pair_scores.get(pair_key(*added), 0.0) if len(added) == 2 else 0.0,
                    "uses_cross_encoder": False,
                    "uses_missing_hop_estimator": False,
                    "uses_document_opportunity_model": False,
                },
            }
        )
    rows.append(
        {
            "query_id": query_id,
            "question": query["question"],
            "action_id": f"{query_id}::v5lite::{variant}::fallback",
            "outer_fold": fold_id,
            "variant": variant,
            "action_family": "fallback",
            "action_name": "baseline_fallback",
            "context_doc_ids": baseline,
            "context_titles": [docs[doc_id]["title"] for doc_id in baseline],
            "context_docs": [docs[doc_id] for doc_id in baseline],
            "added_doc_ids": [],
            "removed_doc_ids": [],
            "ordering": "stable",
            "generator_score": 0.0,
            "is_new_vs_v3_action_table": False,
            "inference_safe_features": {
                "missing_hop_probabilities": {},
                "removal_risk": 0.0,
                "uses_cross_encoder": False,
                "uses_missing_hop_estimator": False,
                "uses_document_opportunity_model": False,
            },
        }
    )
    return rows


def signature(query_id: str, context_ids: list[str]) -> str:
    return f"{query_id}|||{'|||'.join(context_ids)}"


def known_outcomes(v4_common: Any) -> dict[str, dict[str, Any]]:
    known: dict[str, dict[str, Any]] = {}
    current_actions = {
        str(row["action_id"]): row
        for row in read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")
    }
    current_outcomes = {
        str(row["action_id"]): row
        for row in read_jsonl(V4 / "outputs/action_outcomes/v4_action_outputs.jsonl")
    }
    for action_id, action in current_actions.items():
        outcome = current_outcomes[action_id]
        known[signature(str(action["query_id"]), list(action["context_doc_ids"]))] = {
            "query_id": str(action["query_id"]),
            "context_doc_ids": list(action["context_doc_ids"]),
            "prediction": outcome["prediction"],
            "answer_f1": float(outcome["answer_f1"]),
            "title_recall": float(outcome["title_recall"]),
            "title_f1": float(outcome["title_f1"]),
            "answer_title_product": float(outcome["answer_title_product"]),
            "source": "v4_existing",
        }
    for row in v4_common.load_v3_merged_rows():
        key = signature(str(row["query_id"]), list(row["context_doc_ids"]))
        known.setdefault(
            key,
            {
                "query_id": str(row["query_id"]),
                "context_doc_ids": list(row["context_doc_ids"]),
                "prediction": row["prediction"],
                "answer_f1": float(row["answer_f1"]),
                "title_recall": float(row["title_recall"]),
                "title_f1": float(row["title_f1"]),
                "answer_title_product": float(row["answer_title_product"]),
                "source": "v3_existing",
            },
        )
    completion_paths = [
        V4_COMPLETION / "outputs/generator_ablation/reused_context_outcomes.jsonl",
        V4_COMPLETION / "outputs/generator_ablation/reader/pending_context_outcomes.jsonl",
    ]
    for path in completion_paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            known.setdefault(signature(str(row["query_id"]), list(row["context_doc_ids"])), row)
    return known


def lexical_query_cache(question: str, docs: list[dict[str, Any]], baseline_ids: list[str], v4_common: Any) -> dict[str, Any]:
    lexical = v4_common.lexical_doc_features(question, docs, baseline_ids)
    details = {}
    for index, doc in enumerate(docs):
        doc_id = str(doc["doc_id"])
        row = dict(lexical[doc_id])
        row.update(
            {
                "query_doc_cosine": 0.0,
                "cross_encoder_relevance": 0.0,
                "max_baseline_semantic": 0.0,
                "mean_baseline_semantic": 0.0,
                "semantic_novelty": 0.0,
                "source_rank_normalized": float(doc.get("source_rank", index)) / max(1, len(docs) - 1),
            }
        )
        details[doc_id] = row
    candidate_ids = [str(doc["doc_id"]) for doc in docs if str(doc["doc_id"]) not in set(baseline_ids)]
    pair_rows = {}
    by_id = {str(doc["doc_id"]): doc for doc in docs}
    for left_id, right_id in combinations(candidate_ids, 2):
        left_entities = v4_common.capitalized_entities(f"{by_id[left_id]['title']} {by_id[left_id]['text']}")
        right_entities = v4_common.capitalized_entities(f"{by_id[right_id]['title']} {by_id[right_id]['text']}")
        values = [0.0] * 12
        values[8] = v4_common.jaccard(left_entities, right_entities)
        pair_rows[pair_key(left_id, right_id)] = values
    return {
        "question": question,
        "baseline_ids": baseline_ids,
        "candidate_ids": candidate_ids,
        "docs": docs,
        "doc_feature_details": details,
        "pair_features": pair_rows,
    }


def build_revision_holdout(args: argparse.Namespace, v4_common: Any) -> None:
    decision = read_json(OUT / "lite_architecture_decision.json", {})
    variant = decision.get("selected_variant")
    if variant != "lite_lexical_pair":
        raise AssertionError(f"Expected frozen Lite-Lexical-Pair, found {variant}")
    scale_builder = load_module(V4 / "13_build_same_source_scaleup_contexts.py", "v5_revision_holdout_builder")
    previous_cwd = Path.cwd()
    try:
        if str(v4_common.PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(v4_common.PROJECT_ROOT))
        os.chdir(v4_common.PROJECT_ROOT)
        reconstructed = scale_builder.reconstruct_source()
    finally:
        os.chdir(previous_cwd)
    if len(reconstructed) != 7405:
        raise AssertionError(f"Expected 7,405 reconstructed examples, found {len(reconstructed)}")
    existing_dev = read_json(v4_common.source_1000_path())
    existing_holdout = read_json(V4 / "outputs/scaleup/same_source_hotpot_validation_3000.json")
    canonical = scale_builder.canonical_rows
    if canonical(reconstructed[:1000]) != canonical(existing_dev):
        raise AssertionError("The first 1,000 examples do not reproduce the frozen development set")
    if canonical(reconstructed[1000:4000]) != canonical(existing_holdout):
        raise AssertionError("Examples 1,000:4,000 do not reproduce the frozen confirmatory holdout")
    revision = reconstructed[4000:7405]
    revision_ids = {str(row["_id"]) for row in revision}
    prior_ids = {str(row["_id"]) for row in existing_dev + existing_holdout}
    if revision_ids & prior_ids:
        raise AssertionError("Revision holdout overlaps prior outcome sets")
    revision_dir = OUT / "revision_holdout"
    revision_dir.mkdir(parents=True, exist_ok=True)
    write_json(revision_dir / "hotpot_revision_holdout_3405.json", revision)
    selector_v1 = load_project_module(
        v4_common.PROJECT_ROOT / "V7-HP-PAPER/run_support_insertion_selector_v1.py",
        "v5_revision_selector_v1",
        v4_common.PROJECT_ROOT,
    )
    context_rows = [
        scale_builder.baseline_row(item, 4000 + index, selector_v1, selector_v1.READER)
        for index, item in enumerate(revision)
    ]
    write_jsonl(revision_dir / "frozen_baseline_contexts_3405.jsonl", context_rows)
    fold_by_query = {}
    for fold in v4_common.build_folds(revision_ids):
        for query_id in fold["test_query_ids"]:
            fold_by_query[query_id] = int(fold["fold_id"])
    all_actions = []
    for index, row in enumerate(context_rows):
        query_id = str(row["query_id"])
        fold = fold_by_query[query_id]
        model = joblib.load(OUT / "pair_models" / f"fold_{fold}_{variant}.joblib")
        query = lexical_query_cache(
            row["question"], list(row["all_docs"]), list(row["baseline_doc_ids"]), v4_common
        )
        generated = generate_actions(query_id, fold, query, model, variant, args.max_actions)
        for action in generated:
            old_id = str(action["action_id"])
            suffix = old_id.rsplit("::", 1)[-1]
            action["action_id"] = f"{query_id}::v5liteholdout::{variant}::{suffix}"
            action["split"] = "revision_holdout_3405"
        all_actions.extend(generated)
    write_jsonl(revision_dir / "lite_actions_3405.jsonl", all_actions)
    selector = load_project_module(
        V4 / "07_train_nested_selector_v4.py", "v5_revision_lite_selector", v4_common.PROJECT_ROOT
    )
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    baselines = {}
    for action in all_actions:
        query_id = str(action["query_id"])
        if action["action_family"] == "fallback":
            baselines[query_id] = action
        else:
            grouped[int(action["outer_fold"])][query_id].append(action)
    selected_by_query = {}
    fold_records = []
    for fold in range(5):
        bundle = joblib.load(OUT / "selector_models" / f"fold_{fold}_{variant}.joblib")
        query_ids = set(grouped[fold])
        test_rows = [row for query_id in query_ids for row in grouped[fold][query_id]]
        safe = selector.probabilities(bundle["safety_model"], test_rows)
        positive = selector.probabilities(bundle["opportunity_model"], test_rows)
        scored = []
        for row, safe_value, positive_value in zip(test_rows, safe, positive):
            value = dict(row)
            value["pred_answer_safe_prob"] = safe_value
            value["pred_positive_prob"] = positive_value
            scored.append(value)
        selected = selector.select(
            scored,
            query_ids,
            float(bundle["config"]["safe_threshold"]),
            float(bundle["config"]["positive_threshold"]),
            float(bundle["config"]["coverage"]),
        )
        selected_by_query.update(selected)
        fold_records.append(
            {
                "fold": fold,
                "n_queries": len(query_ids),
                "n_selected": len(selected),
                "development_config": bundle["config"],
                "revision_outcomes_used": False,
            }
        )
    selections = []
    for query_id in sorted(baselines):
        selected = selected_by_query.get(query_id)
        current = selected or baselines[query_id]
        selections.append(
            {
                "query_id": query_id,
                "outer_fold": int(current["outer_fold"]),
                "selected": selected is not None,
                "fallback": selected is None,
                "action_id": current["action_id"],
                "action_family": current["action_family"],
                "context_doc_ids": current["context_doc_ids"],
                "context_titles": current["context_titles"],
                "pred_answer_safe_prob": float(current.get("pred_answer_safe_prob", 0.0)),
                "pred_positive_prob": float(current.get("pred_positive_prob", 0.0)),
            }
        )
    write_jsonl(revision_dir / "frozen_lite_selections_3405.jsonl", selections)
    audit = {
        "status": "pass",
        "source_seed": 44,
        "full_validation_size": len(reconstructed),
        "revision_slice": [4000, 7405],
        "revision_size": len(revision),
        "prior_outcome_overlap": 0,
        "development_prefix_exact": True,
        "confirmatory_holdout_prefix_exact": True,
        "selected_variant": variant,
        "architecture_frozen_before_revision_outcomes": True,
        "n_action_rows": len(all_actions),
        "n_selected": sum(row["selected"] for row in selections),
        "coverage": sum(row["selected"] for row in selections) / len(selections),
        "folds": fold_records,
    }
    write_json(revision_dir / "revision_holdout_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def build_revision_full_v4(args: argparse.Namespace, v4_common: Any) -> None:
    revision_dir = OUT / "revision_holdout"
    audit = read_json(revision_dir / "revision_holdout_audit.json", {})
    if audit.get("status") != "pass" or not audit.get("architecture_frozen_before_revision_outcomes"):
        raise AssertionError("Frozen revision holdout audit is required")
    context_rows = read_jsonl(revision_dir / "frozen_baseline_contexts_3405.jsonl")
    generate_module = load_project_module(
        V4 / "14_generate_frozen_scaleup_actions.py", "v5_revision_full_generator", v4_common.PROJECT_ROOT
    )
    train_module = load_project_module(
        V4 / "03_train_semantic_candidate_generator.py", "v5_revision_full_features", v4_common.PROJECT_ROOT
    )
    source, snapshots = generate_module.source_and_snapshots(context_rows)
    cache_path = revision_dir / "full_v4_semantic_feature_cache_3405.joblib"
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
    generator_manifest = read_json(V4 / "outputs/semantic_generator/foldwise_generator_models.json")
    actions = []
    fold_by_query = {}
    for fold in v4_common.build_folds(source):
        fold_id = int(fold["fold_id"])
        model_record = next(row for row in generator_manifest["folds"] if int(row["fold_id"]) == fold_id)
        bundle = joblib.load(model_record["model_path"])
        for query_id in fold["test_query_ids"]:
            fold_by_query[query_id] = fold_id
            generated = generate_module.generate_actions(
                query_id, fold_id, cache["queries"][query_id], bundle, args.max_actions
            )
            for action in generated:
                action["action_id"] = str(action["action_id"]).replace("::v4scale::", "::v4revision::")
                action["split"] = "revision_holdout_3405"
            actions.extend(generated)
    write_jsonl(revision_dir / "full_v4_actions_3405.jsonl", actions)
    selector = load_project_module(
        V4 / "07_train_nested_selector_v4.py", "v5_revision_full_selector", v4_common.PROJECT_ROOT
    )
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    baselines = {}
    for action in actions:
        query_id = str(action["query_id"])
        if action["action_family"] == "fallback":
            baselines[query_id] = action
        else:
            grouped[int(action["outer_fold"])][query_id].append(action)
    selected_by_query = {}
    fold_records = []
    nested = read_json(V4 / "outputs/nested_selector/v4_nested_summary.json")
    for fold in range(5):
        model_bundle = joblib.load(V4 / "outputs/scaleup/selector_models" / f"fold_{fold}_selector.joblib")
        query_ids = set(grouped[fold])
        test_rows = [row for query_id in query_ids for row in grouped[fold][query_id]]
        safe = selector.probabilities(model_bundle["safety_model"], test_rows)
        positive = selector.probabilities(model_bundle["opportunity_model"], test_rows)
        scored = []
        for row, safe_value, positive_value in zip(test_rows, safe, positive):
            value = dict(row)
            value["pred_answer_safe_prob"] = safe_value
            value["pred_positive_prob"] = positive_value
            scored.append(value)
        config_row = next(row for row in nested["folds"] if int(row["outer_fold"]) == fold)["train_selected_config"]
        selected = selector.select(
            scored,
            query_ids,
            float(config_row["safe_threshold"]),
            float(config_row["positive_threshold"]),
            float(config_row["coverage"]),
        )
        selected_by_query.update(selected)
        fold_records.append({"fold": fold, "n_queries": len(query_ids), "n_selected": len(selected), "revision_outcomes_used": False})
    selections = []
    for query_id in sorted(baselines):
        selected = selected_by_query.get(query_id)
        current = selected or baselines[query_id]
        selections.append(
            {
                "query_id": query_id,
                "outer_fold": int(current["outer_fold"]),
                "selected": selected is not None,
                "fallback": selected is None,
                "action_id": current["action_id"],
                "action_family": current["action_family"],
                "context_doc_ids": current["context_doc_ids"],
                "context_titles": current["context_titles"],
            }
        )
    write_jsonl(revision_dir / "full_v4_selections_3405.jsonl", selections)
    manifest = {
        "status": "pass",
        "method": "full_v4",
        "n_queries": len(selections),
        "n_action_rows": len(actions),
        "n_selected": sum(row["selected"] for row in selections),
        "coverage": sum(row["selected"] for row in selections) / len(selections),
        "generator_and_selector_frozen_from_development": True,
        "revision_outcomes_used": False,
        "semantic_feature_cache": str(cache_path),
        "folds": fold_records,
    }
    write_json(revision_dir / "full_v4_generation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_development(args: argparse.Namespace) -> None:
    cfg = config()
    if float(cfg["lite"]["joint_f1_noninferiority_margin"]) != 0.002:
        raise AssertionError("Lite non-inferiority margin changed")
    sys.path.insert(0, str(V4))
    import v4_common

    train_module = load_module(V4 / "03_train_semantic_candidate_generator.py", "v5_lite_train_source")
    cache = joblib.load(V4 / "outputs/semantic_generator/semantic_feature_cache.joblib")
    source = v4_common.load_source_examples()
    targets = train_module.outcome_targets(
        v4_common.grouped_outcomes(v4_common.load_v3_merged_rows()), cache
    )
    folds = v4_common.build_folds(source)
    all_rows: list[dict[str, Any]] = []
    family_counts: dict[str, Counter[str]] = {variant: Counter() for variant in VARIANTS}
    fold_records = []
    model_dir = OUT / "pair_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_ids = list(fold["train_query_ids"])
        test_ids = list(fold["test_query_ids"])
        record = {"fold_id": fold_id, "n_train": len(train_ids), "n_test": len(test_ids), "variants": {}}
        for variant in VARIANTS:
            semantic = variant == "lite_semantic_pair"
            model = fit_pair_model(cache, targets["pair"], train_ids, semantic)
            model_path = model_dir / f"fold_{fold_id}_{variant}.joblib"
            joblib.dump(model, model_path, compress=3)
            record["variants"][variant] = {
                "model_path": str(model_path),
                "semantic_similarity_used": semantic,
                "cross_encoder_used": False,
                "missing_hop_model_used": False,
                "document_opportunity_model_used": False,
                "outer_test_outcomes_used": False,
            }
            for query_id in test_ids:
                generated = generate_actions(
                    query_id, fold_id, cache["queries"][query_id], model, variant, args.max_actions
                )
                all_rows.extend(generated)
                family_counts[variant].update(row["action_family"] for row in generated)
        fold_records.append(record)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "lite_actions_development.jsonl", all_rows)
    known = known_outcomes(v4_common)
    write_jsonl(OUT / "known_context_outcomes.jsonl", known.values())
    pending: dict[str, dict[str, Any]] = {}
    for action in all_rows:
        key = signature(str(action["query_id"]), list(action["context_doc_ids"]))
        if key not in known:
            value = dict(action)
            value["context_signature"] = key
            pending.setdefault(key, value)
    write_jsonl(OUT / "pending_context_actions.jsonl", pending.values())
    audit = {
        "status": "prepared",
        "protocol": "five-fold outer generator training with disjoint outer-test action generation",
        "variants": list(VARIANTS),
        "n_queries": len(source),
        "n_action_rows": len(all_rows),
        "n_known_contexts": len(known),
        "n_pending_contexts": len(pending),
        "joint_f1_noninferiority_margin_frozen": cfg["lite"]["joint_f1_noninferiority_margin"],
        "revision_holdout_outcomes_opened": False,
        "family_counts": {key: dict(value) for key, value in family_counts.items()},
        "folds": fold_records,
    }
    write_json(OUT / "lite_generator_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["development", "revision-holdout", "revision-full-v4"], default="development")
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(V4))
    import v4_common

    if args.stage == "revision-holdout":
        build_revision_holdout(args, v4_common)
    elif args.stage == "revision-full-v4":
        build_revision_full_v4(args, v4_common)
    else:
        build_development(args)


if __name__ == "__main__":
    main()
