#!/usr/bin/env python3
"""Build fully nested V4 generator ablations and identify unevaluated contexts."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from completion_common import ABLATION, V4_ROOT, add_v4_import_path, ensure_layout, load_module, read_json, read_jsonl, write_json, write_jsonl


class SliceModel:
    def __init__(self, model: Any, indices: list[int]) -> None:
        self.model = model
        self.indices = np.asarray(indices, dtype=int)
        self.classes_ = model.classes_

    def predict_proba(self, values: Any) -> Any:
        array = np.asarray(values)
        return self.model.predict_proba(array[:, self.indices])


class ConstantBinaryModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, values: Any) -> np.ndarray:
        n = len(np.asarray(values))
        return np.tile([1.0 - self.probability, self.probability], (n, 1))


class ConstantMissingModel:
    classes_ = np.asarray([
        "missing_bridge", "missing_answer_resolution", "redundant_context", "ordering_problem", "no_intervention_needed"
    ])

    def predict_proba(self, values: Any) -> np.ndarray:
        n = len(np.asarray(values))
        return np.tile([0.25, 0.25, 0.25, 0.25, 0.0], (n, 1))


def fit_binary(train_module: Any, x: list[list[float]], y: list[int], groups: list[str], indices: list[int]) -> SliceModel:
    features = np.asarray(x, dtype=np.float32)[:, indices]
    labels = np.asarray(y, dtype=int)
    c_value, _ = train_module.tune_binary_c(features, labels, groups)
    model = train_module.safe_pipeline(c_value)
    model.fit(features, labels)
    return SliceModel(model, indices)


def fit_multiclass(train_module: Any, x: list[list[float]], y: list[str], groups: list[str], indices: list[int]) -> SliceModel:
    features = np.asarray(x, dtype=np.float32)[:, indices]
    labels = np.asarray(y)
    c_value, _ = train_module.tune_multiclass_c(features, labels, groups)
    model = train_module.safe_pipeline(c_value)
    model.fit(features, labels)
    return SliceModel(model, indices)


def reduced_bundle(
    train_module: Any,
    cache: dict[str, Any],
    targets: dict[str, Any],
    train_ids: list[str],
    query_indices: list[int],
    doc_indices: list[int],
    pair_indices: list[int],
) -> dict[str, Any]:
    query_x = [cache["queries"][qid]["query_features"] for qid in train_ids]
    query_y = [targets["missing"][qid] for qid in train_ids]
    missing_model = fit_multiclass(train_module, query_x, query_y, train_ids, query_indices)
    doc_x: list[list[float]] = []
    doc_y: list[int] = []
    doc_groups: list[str] = []
    pair_x: list[list[float]] = []
    pair_y: list[int] = []
    pair_groups: list[str] = []
    for query_id in train_ids:
        query = cache["queries"][query_id]
        for doc_id in query["candidate_ids"]:
            doc_x.append(query["doc_features"][doc_id])
            doc_y.append(targets["doc"][query_id][doc_id])
            doc_groups.append(query_id)
        for key, label in targets["pair"][query_id].items():
            if key in query["pair_features"]:
                pair_x.append(query["pair_features"][key])
                pair_y.append(label)
                pair_groups.append(query_id)
    doc_model = fit_binary(train_module, doc_x, doc_y, doc_groups, doc_indices)
    pair_model = fit_binary(train_module, pair_x, pair_y, pair_groups, pair_indices) if len(set(pair_y)) >= 2 else None
    return {"missing_model": missing_model, "doc_model": doc_model, "pair_model": pair_model}


def key(query_id: str, context_ids: list[str]) -> str:
    return f"{query_id}|||{'|||'.join(context_ids)}"


def main() -> None:
    ensure_layout()
    add_v4_import_path()
    from semantic_features import DOC_FEATURE_NAMES, PAIR_FEATURE_NAMES, QUERY_FEATURE_NAMES
    from v4_common import build_folds, grouped_outcomes, load_source_examples, load_v3_merged_rows

    train_module = load_module(V4_ROOT / "03_train_semantic_candidate_generator.py", "v4_ablation_train")
    generate_module = load_module(V4_ROOT / "14_generate_frozen_scaleup_actions.py", "v4_ablation_generate")
    cache = joblib.load(V4_ROOT / "outputs/semantic_generator/semantic_feature_cache.joblib")
    source = load_source_examples()
    targets = train_module.outcome_targets(grouped_outcomes(load_v3_merged_rows()), cache)
    manifest = read_json(V4_ROOT / "outputs/semantic_generator/foldwise_generator_models.json")

    def indices(names: list[str], wanted: set[str]) -> list[int]:
        result = [index for index, name in enumerate(names) if name in wanted]
        if not result:
            raise AssertionError(f"Empty feature mask for {wanted}")
        return result

    semantic_doc = {"query_doc_cosine", "cross_encoder_relevance", "max_baseline_semantic", "mean_baseline_semantic", "semantic_novelty"}
    mpnet_doc = {"query_doc_cosine", "max_baseline_semantic", "mean_baseline_semantic", "semantic_novelty"}
    lexical_doc = set(DOC_FEATURE_NAMES) - semantic_doc
    semantic_query = {name for name in QUERY_FEATURE_NAMES if "semantic" in name or "cross" in name or "pair_similarity" in name}
    lexical_query = set(QUERY_FEATURE_NAMES) - semantic_query
    semantic_pair = {name for name in PAIR_FEATURE_NAMES if "semantic" in name or "cross" in name or "cosine" in name or "opportunity_prior" in name}
    lexical_pair = set(PAIR_FEATURE_NAMES) - semantic_pair
    no_mpnet_doc = set(DOC_FEATURE_NAMES) - mpnet_doc
    no_mpnet_query = {name for name in QUERY_FEATURE_NAMES if "semantic" not in name and "pair_similarity" not in name}
    no_mpnet_pair = {name for name in PAIR_FEATURE_NAMES if "semantic" not in name and "cosine" not in name and "opportunity_prior" not in name}
    no_cross_doc = set(DOC_FEATURE_NAMES) - {"cross_encoder_relevance"}
    no_cross_query = {name for name in QUERY_FEATURE_NAMES if "cross" not in name}
    no_cross_pair = {name for name in PAIR_FEATURE_NAMES if "cross" not in name and "opportunity_prior" not in name}

    variants = {
        "full": {"mode": "full"},
        "without_missing_hop_estimator": {"mode": "constant_missing"},
        "without_mpnet_features": {"mode": "retrain", "q": no_mpnet_query, "d": no_mpnet_doc, "p": no_mpnet_pair},
        "without_cross_encoder_features": {"mode": "retrain", "q": no_cross_query, "d": no_cross_doc, "p": no_cross_pair},
        "without_semantic_document_model": {"mode": "constant_doc"},
        "without_pair_complementarity": {"mode": "no_pair"},
        "without_two_document_actions": {"mode": "filter", "families": {"semantic_two_document_chain"}},
        "without_anchor_preservation": {"mode": "filter", "families": {"anchor_preserving_replacement", "answer_anchor_first_reorder"}},
        "without_redundancy_actions": {"mode": "filter", "families": {"redundancy_replacement"}},
        "lexical_only_generator": {"mode": "retrain", "q": lexical_query, "d": lexical_doc, "p": lexical_pair},
        "semantic_only_generator": {"mode": "retrain", "q": semantic_query, "d": semantic_doc, "p": semantic_pair},
    }

    all_rows: list[dict[str, Any]] = []
    variant_counts: dict[str, Counter[str]] = {name: Counter() for name in variants}
    for fold in build_folds(source):
        fold_id = int(fold["fold_id"])
        record = next(value for value in manifest["folds"] if int(value["fold_id"]) == fold_id)
        full_bundle = joblib.load(record["model_path"])
        train_ids = list(fold["train_query_ids"])
        bundles: dict[str, dict[str, Any]] = {}
        for name, config in variants.items():
            mode = config["mode"]
            if mode in {"full", "filter"}:
                bundles[name] = full_bundle
            elif mode == "constant_missing":
                bundle = copy.copy(full_bundle)
                bundle["missing_model"] = ConstantMissingModel()
                bundles[name] = bundle
            elif mode == "constant_doc":
                bundle = copy.copy(full_bundle)
                prevalence = float(record["doc_positive_rows"]) / max(1, int(record["doc_training_rows"]))
                bundle["doc_model"] = ConstantBinaryModel(prevalence)
                bundles[name] = bundle
            elif mode == "no_pair":
                bundle = copy.copy(full_bundle)
                bundle["pair_model"] = None
                bundles[name] = bundle
            else:
                bundles[name] = reduced_bundle(
                    train_module,
                    cache,
                    targets,
                    train_ids,
                    indices(QUERY_FEATURE_NAMES, config["q"]),
                    indices(DOC_FEATURE_NAMES, config["d"]),
                    indices(PAIR_FEATURE_NAMES, config["p"]),
                )
        for query_id in fold["test_query_ids"]:
            for name, config in variants.items():
                generated = generate_module.generate_actions(query_id, fold_id, cache["queries"][query_id], bundles[name], 8)
                excluded = config.get("families", set())
                generated = [row for row in generated if row["action_family"] == "fallback" or row["action_family"] not in excluded]
                for index, row in enumerate(generated):
                    suffix = "fallback" if row["action_family"] == "fallback" else f"{index:02d}"
                    row["variant"] = name
                    row["action_id"] = f"{query_id}::ablation::{name}::{suffix}"
                    all_rows.append(row)
                    variant_counts[name][row["action_family"]] += 1

    actions_path = ABLATION / "generator_ablation_actions.jsonl"
    write_jsonl(actions_path, all_rows)

    known: dict[str, dict[str, Any]] = {}
    current_actions = {str(row["action_id"]): row for row in read_jsonl(V4_ROOT / "outputs/generated_actions/v4_outer_test_actions.jsonl")}
    current_outcomes = {str(row["action_id"]): row for row in read_jsonl(V4_ROOT / "outputs/action_outcomes/v4_action_outputs.jsonl")}
    for action_id, action in current_actions.items():
        outcome = current_outcomes[action_id]
        known[key(str(action["query_id"]), list(action["context_doc_ids"]))] = {
            "query_id": str(action["query_id"]),
            "context_doc_ids": list(action["context_doc_ids"]),
            "prediction": outcome["prediction"],
            "answer_f1": float(outcome["answer_f1"]),
            "title_recall": float(outcome["title_recall"]),
            "title_f1": float(outcome["title_f1"]),
            "answer_title_product": float(outcome["answer_title_product"]),
            "source": "v4_existing",
        }
    for row in load_v3_merged_rows():
        signature = key(str(row["query_id"]), list(row["context_doc_ids"]))
        known.setdefault(signature, {
            "query_id": str(row["query_id"]),
            "context_doc_ids": list(row["context_doc_ids"]),
            "prediction": row["prediction"],
            "answer_f1": float(row["answer_f1"]),
            "title_recall": float(row["title_recall"]),
            "title_f1": float(row["title_f1"]),
            "answer_title_product": float(row["answer_title_product"]),
            "source": "v3_existing",
        })
    write_jsonl(ABLATION / "reused_context_outcomes.jsonl", known.values())

    pending_by_key: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        signature = key(str(row["query_id"]), list(row["context_doc_ids"]))
        if signature not in known and signature not in pending_by_key:
            pending = dict(row)
            pending["context_signature"] = signature
            pending_by_key[signature] = pending
    pending_path = ABLATION / "pending_context_actions.jsonl"
    write_jsonl(pending_path, pending_by_key.values())
    audit = {
        "status": "prepared",
        "protocol": "fully nested outer-train component ablations; structural removals use the same outer-test generator",
        "variants": list(variants),
        "n_variant_action_rows": len(all_rows),
        "n_known_unique_contexts": len(known),
        "n_pending_unique_contexts": len(pending_by_key),
        "target_query_reader_outcomes_used_for_generation": False,
        "holdout_used": False,
        "variant_family_counts": {name: dict(counts) for name, counts in variant_counts.items()},
    }
    write_json(ABLATION / "generator_ablation_preparation_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
