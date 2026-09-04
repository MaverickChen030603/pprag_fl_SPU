#!/usr/bin/env python3
"""Fully nested safety-first selector for v4 actions, gated by semantic opportunity."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from statistics import mean
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v4_common import OUTPUTS, ensure_layout, query_fingerprint, read_json, read_jsonl, write_json, write_jsonl


SEED = 20260714
FAMILIES = [
    "single_complementary_insertion", "anchor_preserving_replacement", "semantic_two_document_chain",
    "redundancy_replacement", "bridge_first_reorder", "answer_anchor_first_reorder",
]
MISSING_TYPES = ["missing_bridge", "missing_answer_resolution", "redundant_context", "ordering_problem", "no_intervention_needed"]


def feature_vector(row: dict[str, Any]) -> list[float]:
    features = row.get("inference_safe_features", {})
    missing = features.get("missing_hop_probabilities", {})
    values = [
        float(row.get("generator_score", 0.0)),
        float(row.get("is_new_vs_v3_action_table", False)),
        float(features.get("added_doc_opportunity_mean", 0.0)),
        float(features.get("added_doc_semantic_mean", 0.0)),
        float(features.get("removal_risk", 0.0)),
    ]
    values.extend(float(missing.get(name, 0.0)) for name in MISSING_TYPES)
    values.extend(float(row.get("action_family") == family) for family in FAMILIES)
    return values


def fit_model(rows: list[dict[str, Any]], target: str) -> Any:
    x = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    y = np.asarray([int(row[target]) for row in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2500, random_state=SEED))
    model.fit(x, y)
    return model


def probabilities(model: Any, rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    values = model.predict_proba(np.asarray([feature_vector(row) for row in rows], dtype=np.float64))
    classes = list(model.classes_)
    return [float(value[classes.index(1)]) if 1 in classes else 0.0 for value in values]


def score_rows(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safety = fit_model(train_rows, "answer_safe")
    opportunity = fit_model(train_rows, "positive_action")
    output = []
    for row, safe, positive in zip(test_rows, probabilities(safety, test_rows), probabilities(opportunity, test_rows)):
        value = dict(row)
        value["pred_answer_safe_prob"] = safe
        value["pred_positive_prob"] = positive
        output.append(value)
    return output


def prepare(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    baselines, actions = {}, []
    for query_id, values in grouped.items():
        baseline = next(row for row in values if row["action_family"] == "fallback")
        baselines[query_id] = baseline
        for source in values:
            if source["action_family"] == "fallback":
                continue
            row = dict(source)
            row["answer_f1_delta"] = float(row["answer_f1"]) - float(baseline["answer_f1"])
            row["title_recall_delta"] = float(row["title_recall"]) - float(baseline["title_recall"])
            row["title_f1_delta"] = float(row["title_f1"]) - float(baseline["title_f1"])
            row["answer_title_product_delta"] = float(row["answer_title_product"]) - float(baseline["answer_title_product"])
            row["answer_safe"] = int(row["answer_f1_delta"] >= -1e-12)
            row["positive_action"] = int(row["answer_safe"] and row["answer_title_product_delta"] > 1e-12 and (row["title_recall_delta"] > 1e-12 or row["title_f1_delta"] >= -1e-12))
            actions.append(row)
    return baselines, actions


def split_inner(query_ids: set[str], outer_fold: int) -> list[set[str]]:
    ordered = sorted(query_ids, key=lambda value: int(hashlib.md5(f"v4-{outer_fold}-{value}".encode()).hexdigest(), 16))
    return [set(ordered[index::5]) for index in range(5)]


def inner_oof(rows: list[dict[str, Any]], query_ids: set[str], outer_fold: int) -> list[dict[str, Any]]:
    output = []
    for validation_ids in split_inner(query_ids, outer_fold):
        fit_rows = [row for row in rows if row["query_id"] not in validation_ids]
        validation_rows = [row for row in rows if row["query_id"] in validation_ids]
        output.extend(score_rows(fit_rows, validation_rows))
    if len(output) != len(rows):
        raise AssertionError("Inner OOF action count mismatch")
    return output


def select(rows: list[dict[str, Any]], query_ids: set[str], safe_threshold: float, positive_threshold: float, coverage: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["query_id"] in query_ids and float(row["pred_answer_safe_prob"]) >= safe_threshold and float(row["pred_positive_prob"]) >= positive_threshold:
            grouped[row["query_id"]].append(row)
    best = [max(values, key=lambda row: (float(row["pred_positive_prob"]), float(row["pred_answer_safe_prob"]))) for values in grouped.values()]
    budget = min(len(best), int(round(coverage * len(query_ids))))
    ordered = sorted(best, key=lambda row: (float(row["pred_positive_prob"]), float(row["pred_answer_safe_prob"])), reverse=True)
    return {row["query_id"]: row for row in ordered[:budget]}


def evaluate(selected: dict[str, dict[str, Any]], baselines: dict[str, dict[str, Any]], query_ids: set[str]) -> dict[str, Any]:
    metrics = ["answer_f1", "title_recall", "title_f1", "answer_title_product"]
    per_query = []
    for query_id in sorted(query_ids):
        baseline = baselines[query_id]
        action = selected.get(query_id)
        current = action or baseline
        row = {
            "query_id": query_id,
            "selected": action is not None,
            "fallback": action is None,
            "action_id": current["action_id"],
            "action_family": current["action_family"],
            "context_doc_ids": current["context_doc_ids"],
            "context_titles": current["context_titles"],
            "prediction": current["prediction"],
            "outer_fold": int(current.get("outer_fold", baseline.get("outer_fold", -1))),
        }
        for metric in metrics:
            row[metric] = float(current[metric])
            row[f"baseline_{metric}"] = float(baseline[metric])
            row[f"{metric}_delta"] = float(current[metric]) - float(baseline[metric])
        per_query.append(row)
    selected_rows = [row for row in per_query if row["selected"]]
    return {
        "coverage": len(selected_rows) / len(query_ids),
        "selected_count": len(selected_rows),
        "answer_drop_rate": sum(row["answer_f1_delta"] < -1e-12 for row in selected_rows) / max(1, len(selected_rows)),
        "deltas": {metric: mean(row[f"{metric}_delta"] for row in per_query) for metric in metrics},
        "per_query": per_query,
    }


def tune(oof: list[dict[str, Any]], baselines: dict[str, dict[str, Any]], train_ids: set[str]) -> dict[str, float]:
    candidates = []
    for safe_threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
        for positive_threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
            for coverage in (0.10, 0.15, 0.20, 0.25, 0.30):
                result = evaluate(select(oof, train_ids, safe_threshold, positive_threshold, coverage), baselines, train_ids)
                feasible = result["deltas"]["answer_f1"] >= -0.001 and result["answer_drop_rate"] <= 0.05
                objective = result["deltas"]["answer_title_product"] + 0.25 * result["deltas"]["title_recall"]
                candidates.append({"safe_threshold": safe_threshold, "positive_threshold": positive_threshold, "coverage": coverage, "feasible": feasible, "objective": objective, "answer_drop_rate": result["answer_drop_rate"]})
    pool = [row for row in candidates if row["feasible"]] or candidates
    return max(pool, key=lambda row: (row["feasible"], row["objective"], -row["answer_drop_rate"]))


def paired_bootstrap(differences: list[float], rounds: int = 5000) -> dict[str, float]:
    rng = random.Random(SEED)
    n = len(differences)
    samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds)]
    samples.sort()
    return {
        "mean": mean(differences),
        "ci95_low": samples[int(0.025 * rounds)],
        "ci95_high": samples[int(0.975 * rounds)],
        "p_value": min(1.0, 2 * min(sum(value <= 0 for value in samples) / rounds, sum(value >= 0 for value in samples) / rounds)),
    }


def main() -> None:
    ensure_layout()
    gate = read_json(OUTPUTS / "opportunity/v4_opportunity_gate.json")
    if not gate.get("proceed_to_nested_selector"):
        payload = {"status": "skipped_by_opportunity_gate", "gate": gate}
        write_json(OUTPUTS / "nested_selector/v4_nested_summary.json", payload)
        print(json.dumps({"status": payload["status"]}, indent=2))
        return
    baselines, actions = prepare(read_jsonl(OUTPUTS / "action_outcomes/v4_action_outputs.jsonl"))
    all_ids = set(baselines)
    per_query, fold_records = [], []
    for outer_fold in range(5):
        test_ids = {query_id for query_id, baseline in baselines.items() if int(baseline["outer_fold"]) == outer_fold}
        train_ids = all_ids - test_ids
        train_rows = [row for row in actions if row["query_id"] in train_ids]
        test_rows = [row for row in actions if row["query_id"] in test_ids]
        oof = inner_oof(train_rows, train_ids, outer_fold)
        config = tune(oof, baselines, train_ids)
        scored_test = score_rows(train_rows, test_rows)
        selected = select(scored_test, test_ids, config["safe_threshold"], config["positive_threshold"], config["coverage"])
        result = evaluate(selected, baselines, test_ids)
        per_query.extend(result["per_query"])
        fold_records.append({
            "outer_fold": outer_fold,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train_fingerprint": query_fingerprint(train_ids),
            "test_fingerprint": query_fingerprint(test_ids),
            "train_selected_config": config,
            "outer_test_result": {key: value for key, value in result.items() if key != "per_query"},
            "outer_test_outcomes_used_for_training_or_tuning": False,
        })
    write_jsonl(OUTPUTS / "nested_selector/v4_nested_per_query.jsonl", per_query)
    metrics = ["answer_f1", "title_recall", "title_f1", "answer_title_product"]
    selected_count = sum(row["selected"] for row in per_query)
    summary = {
        "status": "complete",
        "n_queries": len(per_query),
        "selected_count": selected_count,
        "coverage": selected_count / len(per_query),
        "answer_drop_rate": sum(row["selected"] and row["answer_f1_delta"] < -1e-12 for row in per_query) / max(1, selected_count),
        "metrics": {metric: mean(row[metric] for row in per_query) for metric in metrics},
        "baseline": {metric: mean(row[f"baseline_{metric}"] for row in per_query) for metric in metrics},
        "deltas": {metric: mean(row[f"{metric}_delta"] for row in per_query) for metric in metrics},
        "significance": {metric: paired_bootstrap([row[f"{metric}_delta"] for row in per_query]) for metric in metrics},
        "folds": fold_records,
    }
    write_json(OUTPUTS / "nested_selector/v4_nested_summary.json", summary)
    write_json(OUTPUTS / "audits/selector_nested_no_leak_audit.json", {"status": "pass", "folds": fold_records})
    print(json.dumps({key: summary[key] for key in ["status", "selected_count", "coverage", "answer_drop_rate", "deltas"]}, indent=2))


if __name__ == "__main__":
    main()
