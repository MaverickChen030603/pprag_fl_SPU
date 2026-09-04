#!/usr/bin/env python3
"""Fully nested v3 selector with train-only safety, threshold, and coverage tuning."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v3_common import HERE, OUTPUTS, ensure_layout, markdown_table, paired_bootstrap, read_json, read_jsonl, write_json, write_jsonl


SEED = 20260713
FAMILIES = [
    "anchor_preserving_tail_replacement",
    "bridge_aware_complementary_insertion",
    "bounded_two_document_chain",
    "redundancy_aware_replacement",
    "joint_reorder_and_insert",
]


def query_hash(query_id: str, salt: str = "") -> int:
    return int(hashlib.md5((salt + query_id).encode("utf-8")).hexdigest(), 16)


def split_ids(query_ids: set[str], folds: int, salt: str) -> list[set[str]]:
    ordered = sorted(query_ids, key=lambda value: query_hash(value, salt))
    return [set(ordered[index::folds]) for index in range(folds)]


def fingerprint(query_ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(query_ids)).encode("utf-8")).hexdigest()


def prepare(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    baselines: dict[str, dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []
    for query_id, values in by_query.items():
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


def feature_vector(row: dict[str, Any]) -> list[float]:
    features = row.get("inference_safe_features", {})
    names = [
        "added_bm25_mean", "added_query_overlap_mean", "added_entity_overlap_mean",
        "added_title_overlap_mean", "added_bridge_connection_mean",
        "added_novel_entity_ratio_mean", "added_redundancy_mean",
        "added_anchor_proxy_mean", "removed_anchor_proxy_mean", "displaced_score_margin",
        "preserved_prefix_length", "num_added_docs", "num_removed_docs",
        "ordering_anchor_first", "ordering_bridge_first",
    ]
    values = [float(features.get(name, 0.0) or 0.0) for name in names]
    values.extend(float(row.get("action_family") == family) for family in FAMILIES)
    values.extend([
        float("keep_top3" in str(row.get("action_name", ""))),
        float("keep_top2" in str(row.get("action_name", ""))),
        float("chain" in str(row.get("action_name", ""))),
        float(row.get("safety_prior") == "strict"),
    ])
    return values


def fit_classifier(rows: list[dict[str, Any]], target: str):
    x = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    y = np.asarray([int(row[target]) for row in rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
        model.fit(x, y)
        return model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    model.fit(x, y)
    return model


def probability(model: Any, rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    values = model.predict_proba(np.asarray([feature_vector(row) for row in rows], dtype=np.float64))
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0] * len(rows)
    return [float(row[classes.index(1)]) for row in values]


def add_predictions(rows: list[dict[str, Any]], safe: list[float], positive: list[float], source: str) -> list[dict[str, Any]]:
    out = []
    for row, p_safe, p_positive in zip(rows, safe, positive):
        value = dict(row)
        value["pred_answer_safe_prob"] = p_safe
        value["pred_positive_prob"] = p_positive
        value["nested_prediction_source"] = source
        out.append(value)
    return out


def nested_predictions(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], train_ids: set[str], outer_fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inner_sets = split_ids(train_ids, 5, f"outer-{outer_fold}-inner")
    oof: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for inner_fold, validation_ids in enumerate(inner_sets):
        fit_rows = [row for row in train_rows if row["query_id"] not in validation_ids]
        validation_rows = [row for row in train_rows if row["query_id"] in validation_ids]
        safe_model = fit_classifier(fit_rows, "answer_safe")
        positive_model = fit_classifier(fit_rows, "positive_action")
        oof.extend(add_predictions(validation_rows, probability(safe_model, validation_rows), probability(positive_model, validation_rows), f"outer_{outer_fold}_inner_oof_{inner_fold}"))
        fit_ids = train_ids - validation_ids
        records.append({
            "inner_fold": inner_fold,
            "n_train_queries": len(fit_ids),
            "n_validation_queries": len(validation_ids),
            "query_overlap": len(fit_ids & validation_ids),
            "train_fingerprint": fingerprint(fit_ids),
            "validation_fingerprint": fingerprint(validation_ids),
        })
    safe_model = fit_classifier(train_rows, "answer_safe")
    positive_model = fit_classifier(train_rows, "positive_action")
    test = add_predictions(test_rows, probability(safe_model, test_rows), probability(positive_model, test_rows), f"outer_{outer_fold}_train_only")
    if len(oof) != len(train_rows):
        raise AssertionError(f"Outer fold {outer_fold}: inner OOF action count mismatch")
    return oof, test, records


def choose(rows: list[dict[str, Any]], query_ids: set[str], threshold: float, coverage: float, rank_key: str = "pred_positive_prob") -> dict[str, dict[str, Any]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["query_id"] in query_ids and float(row.get("pred_answer_safe_prob", 0.0)) >= threshold:
            by_query[row["query_id"]].append(row)
    best = {query_id: max(values, key=lambda row: (float(row.get(rank_key, 0.0)), float(row.get("pred_answer_safe_prob", 0.0)))) for query_id, values in by_query.items()}
    budget = min(len(best), int(round(coverage * len(query_ids))))
    ordered = sorted(best.values(), key=lambda row: (float(row.get(rank_key, 0.0)), float(row.get("pred_answer_safe_prob", 0.0)), -query_hash(row["query_id"], "budget")), reverse=True)
    return {row["query_id"]: row for row in ordered[:budget]}


def evaluate(selected: dict[str, dict[str, Any]], baselines: dict[str, dict[str, Any]], query_ids: set[str]) -> dict[str, Any]:
    metric_names = ["answer_f1", "title_recall", "title_f1", "answer_title_product"]
    selected_metrics = Counter()
    baseline_metrics = Counter()
    answer_drops = 0
    per_query: list[dict[str, Any]] = []
    for query_id in sorted(query_ids):
        baseline = baselines[query_id]
        action = selected.get(query_id)
        current = action or baseline
        row = {
            "query_id": query_id,
            "selected": action is not None,
            "fallback": action is None,
            "action_id": action["action_id"] if action else baseline["action_id"],
            "action_family": action["action_family"] if action else "fallback",
            "action_name": action["action_name"] if action else "baseline_fallback",
            "context_doc_ids": current["context_doc_ids"],
            "context_titles": current["context_titles"],
            "prediction": current["prediction"],
        }
        for metric in metric_names:
            selected_metrics[metric] += float(current[metric])
            baseline_metrics[metric] += float(baseline[metric])
            row[metric] = float(current[metric])
            row[f"baseline_{metric}"] = float(baseline[metric])
            row[f"{metric}_delta"] = float(current[metric]) - float(baseline[metric])
        if action:
            answer_drops += int(row["answer_f1_delta"] < -1e-12)
            row["pred_answer_safe_prob"] = float(action.get("pred_answer_safe_prob", 0.0))
            row["pred_positive_prob"] = float(action.get("pred_positive_prob", 0.0))
            row["inference_safe_features"] = action.get("inference_safe_features", {})
        per_query.append(row)
    n = len(query_ids)
    selected_count = len(selected)
    metrics = {metric: selected_metrics[metric] / n for metric in metric_names}
    baseline_mean = {metric: baseline_metrics[metric] / n for metric in metric_names}
    return {
        "n_queries": n,
        "selected_count": selected_count,
        "fallback_count": n - selected_count,
        "coverage": selected_count / n,
        "selected_answer_drop_count": answer_drops,
        "selected_answer_drop_rate": answer_drops / selected_count if selected_count else 0.0,
        "metrics": metrics,
        "baseline": baseline_mean,
        "deltas": {metric: metrics[metric] - baseline_mean[metric] for metric in metric_names},
        "per_query": per_query,
    }


def tune(oof_rows: list[dict[str, Any]], baselines: dict[str, dict[str, Any]], train_ids: set[str], epsilon: float, risk_budget: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        for coverage in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            result = evaluate(choose(oof_rows, train_ids, threshold, coverage), baselines, train_ids)
            feasible = result["deltas"]["answer_f1"] >= -epsilon and result["selected_answer_drop_rate"] <= risk_budget
            objective = result["deltas"]["answer_title_product"] + 0.25 * result["deltas"]["title_recall"] + 0.15 * result["deltas"]["title_f1"]
            candidates.append({
                "threshold": threshold,
                "target_coverage": coverage,
                "realized_coverage": result["coverage"],
                "answer_f1_delta": result["deltas"]["answer_f1"],
                "title_recall_delta": result["deltas"]["title_recall"],
                "title_f1_delta": result["deltas"]["title_f1"],
                "product_delta": result["deltas"]["answer_title_product"],
                "selected_answer_drop_rate": result["selected_answer_drop_rate"],
                "objective": objective,
                "feasible": feasible,
            })
    feasible = [row for row in candidates if row["feasible"]]
    pool = feasible or candidates
    best = max(pool, key=lambda row: (row["feasible"], row["objective"], row["answer_f1_delta"], -row["selected_answer_drop_rate"]))
    return best, candidates


def summarize(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["answer_f1", "title_recall", "title_f1", "answer_title_product"]
    n = len(per_query)
    summary = {
        "status": "complete",
        "n_queries": n,
        "outer_folds": 5,
        "selected_count": sum(row["selected"] for row in per_query),
        "fallback_count": sum(row["fallback"] for row in per_query),
        "selected_answer_drop_count": sum(row["selected"] and row["answer_f1_delta"] < -1e-12 for row in per_query),
    }
    summary["coverage"] = summary["selected_count"] / n
    summary["selected_answer_drop_rate"] = summary["selected_answer_drop_count"] / summary["selected_count"] if summary["selected_count"] else 0.0
    summary["metrics"] = {metric: mean(row[metric] for row in per_query) for metric in metrics}
    summary["baseline"] = {metric: mean(row[f"baseline_{metric}"] for row in per_query) for metric in metrics}
    summary["deltas"] = {metric: mean(row[f"{metric}_delta"] for row in per_query) for metric in metrics}
    return summary


def matched_baseline(name: str, scored_rows: list[dict[str, Any]], baselines: dict[str, dict[str, Any]], query_ids: set[str], count: int) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_query[row["query_id"]].append(row)
    chosen: list[tuple[float, dict[str, Any]]] = []
    for query_id, values in by_query.items():
        if name == "bm25_reranking":
            best = max(values, key=lambda row: float(row["inference_safe_features"].get("added_bm25_mean", 0.0)))
            score = float(best["inference_safe_features"].get("added_bm25_mean", 0.0))
        elif name == "support_first_selector":
            best = max(values, key=lambda row: float(row["inference_safe_features"].get("added_bridge_connection_mean", 0.0)) + float(row["inference_safe_features"].get("added_bm25_mean", 0.0)))
            score = float(best["inference_safe_features"].get("added_bridge_connection_mean", 0.0)) + float(best["inference_safe_features"].get("added_bm25_mean", 0.0))
        elif name == "answer_safety_only_selector":
            best = max(values, key=lambda row: float(row.get("pred_answer_safe_prob", 0.0)))
            score = float(best.get("pred_answer_safe_prob", 0.0))
        elif name == "random_effective_action":
            best = min(values, key=lambda row: query_hash(row["action_id"], "random-action"))
            score = float(-query_hash(query_id, "random-query"))
        elif name == "oracle_action_upper_bound":
            best = max(values, key=lambda row: (float(row["answer_safe"]), float(row["answer_title_product_delta"]), float(row["title_recall_delta"])))
            score = float(best["answer_title_product_delta"])
        else:
            raise ValueError(name)
        chosen.append((score, best))
    if name == "oracle_action_upper_bound":
        selected = {row["query_id"]: row for _, row in chosen if row["answer_safe"] and row["answer_title_product_delta"] > 0}
    else:
        selected = {row["query_id"]: row for _, row in sorted(chosen, key=lambda value: value[0], reverse=True)[:count]}
    result = evaluate(selected, baselines, query_ids)
    result.pop("per_query")
    return result


def main() -> None:
    ensure_layout()
    config = read_json(HERE / "configs/experiment_v3.json")
    rows = read_jsonl(OUTPUTS / "action_outcomes/v3_action_reader_outputs.jsonl")
    outcome_summary = read_json(OUTPUTS / "action_outcomes/v3_action_outcome_summary.json")
    if outcome_summary.get("status") != "complete":
        raise RuntimeError("Stage 3 reader outcomes are incomplete")
    if not outcome_summary.get("proceed_to_nested_selector"):
        write_json(OUTPUTS / "nested_selector/v3_nested_summary.json", {"status": "skipped_by_opportunity_gate", "positive_query_coverage": outcome_summary.get("positive_query_coverage")})
        print("Stage 4 skipped by the pre-registered opportunity gate")
        return
    baselines, actions = prepare(rows)
    query_ids = set(baselines)
    outer_sets = split_ids(query_ids, 5, "outer-v3")
    per_query: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    all_scored_test: list[dict[str, Any]] = []

    for outer_fold, test_ids in enumerate(outer_sets):
        train_ids = query_ids - test_ids
        train_rows = [row for row in actions if row["query_id"] in train_ids]
        test_rows = [row for row in actions if row["query_id"] in test_ids]
        oof, scored_test, inner_records = nested_predictions(train_rows, test_rows, train_ids, outer_fold)
        best, candidates = tune(oof, baselines, train_ids, float(config["answer_delta_epsilon"]), float(config["selected_answer_drop_risk_budget"]))
        selected = choose(scored_test, test_ids, float(best["threshold"]), float(best["target_coverage"]))
        result = evaluate(selected, baselines, test_ids)
        for row in result["per_query"]:
            row["outer_fold"] = outer_fold
            row["train_selected_threshold"] = best["threshold"]
            row["train_selected_target_coverage"] = best["target_coverage"]
        per_query.extend(result["per_query"])
        all_scored_test.extend(scored_test)
        fold_summaries.append({"outer_fold": outer_fold, "train_config": best, "outer_test": {key: value for key, value in result.items() if key != "per_query"}})
        fold_audits.append({
            "outer_fold": outer_fold,
            "n_train_queries": len(train_ids),
            "n_test_queries": len(test_ids),
            "outer_query_overlap": len(train_ids & test_ids),
            "train_fingerprint": fingerprint(train_ids),
            "test_fingerprint": fingerprint(test_ids),
            "inner_folds": inner_records,
            "outer_test_outcomes_used_for_training": False,
            "outer_test_outcomes_used_for_threshold_selection": False,
            "outer_test_outcomes_used_for_coverage_selection": False,
        })
        for candidate in candidates:
            value = dict(candidate)
            value["outer_fold"] = outer_fold
            risk_rows.append(value)

    summary = summarize(per_query)
    summary["protocol"] = {
        "fully_nested": True,
        "inner_oof_nuisance_and_selector_predictions": True,
        "train_only_threshold_selection": True,
        "train_only_coverage_selection": True,
        "answer_delta_epsilon": config["answer_delta_epsilon"],
        "selected_answer_drop_risk_budget": config["selected_answer_drop_risk_budget"],
    }
    significance = {metric: paired_bootstrap([row[f"{metric}_delta"] for row in per_query], seed=SEED + index) for index, metric in enumerate(["answer_f1", "title_recall", "title_f1", "answer_title_product"])}
    matched_count = summary["selected_count"]
    baselines_summary = {
        name: matched_baseline(name, all_scored_test, baselines, query_ids, matched_count)
        for name in ["bm25_reranking", "support_first_selector", "answer_safety_only_selector", "random_effective_action", "oracle_action_upper_bound"]
    }
    v2 = read_json(HERE.parent / "submission_revision_v2/nested_final_1000_summary.json")
    baselines_summary["v2_constrained_selector"] = {"source": "frozen_submission_revision_v2", "summary": v2}
    baselines_summary["v3_candidate_generator_constrained_selector"] = summary

    write_json(OUTPUTS / "nested_selector/v3_nested_summary.json", summary)
    write_jsonl(OUTPUTS / "nested_selector/v3_nested_per_query.jsonl", per_query)
    write_json(OUTPUTS / "nested_selector/v3_nested_significance.json", significance)
    write_json(OUTPUTS / "nested_selector/v3_nested_fold_summary.json", fold_summaries)
    write_json(OUTPUTS / "audits/v3_nested_no_leak_audit.json", {"status": "pass", "folds": fold_audits})
    write_json(OUTPUTS / "nested_selector/strong_baseline_summary.json", baselines_summary)

    coverage_table_rows: list[list[Any]] = []
    aggregated: list[dict[str, Any]] = []
    for coverage in config["coverage_grid"]:
        values = [row for row in risk_rows if abs(float(row["target_coverage"]) - float(coverage)) < 1e-9]
        best_by_fold = [max([row for row in values if row["outer_fold"] == fold], key=lambda row: (row["feasible"], row["objective"])) for fold in range(5)]
        record = {
            "coverage": coverage,
            "mean_answer_f1_delta": mean(row["answer_f1_delta"] for row in best_by_fold),
            "mean_product_delta": mean(row["product_delta"] for row in best_by_fold),
            "mean_title_recall_delta": mean(row["title_recall_delta"] for row in best_by_fold),
            "mean_answer_drop_rate": mean(row["selected_answer_drop_rate"] for row in best_by_fold),
            "feasible_folds": sum(row["feasible"] for row in best_by_fold),
        }
        aggregated.append(record)
        coverage_table_rows.append([f"{coverage:.1f}", f"{record['mean_answer_f1_delta']:+.4f}", f"{record['mean_product_delta']:+.4f}", f"{record['mean_title_recall_delta']:+.4f}", f"{record['mean_answer_drop_rate']:.3f}", record["feasible_folds"]])
    (OUTPUTS / "tables/v3_risk_coverage_table.md").write_text("# v3 Train-Only Risk-Coverage Table\n\n" + markdown_table(["Target coverage", "OOF answer delta", "OOF product delta", "OOF title recall delta", "OOF answer-drop", "Feasible folds"], coverage_table_rows) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        x = [row["coverage"] for row in aggregated]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(x, [row["mean_product_delta"] for row in aggregated], marker="o", label="Product delta")
        ax.plot(x, [row["mean_answer_f1_delta"] for row in aggregated], marker="s", label="Answer F1 delta")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Train-only target coverage")
        ax.set_ylabel("Mean outer-train OOF delta")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUTS / "figures/v3_risk_coverage_curve.pdf")
        plt.close(fig)
    except Exception as exc:
        write_json(OUTPUTS / "figures/v3_risk_coverage_curve_error.json", {"error": repr(exc)})
    print(json.dumps({"deltas": summary["deltas"], "coverage": summary["coverage"], "answer_drop_rate": summary["selected_answer_drop_rate"]}, indent=2))


if __name__ == "__main__":
    main()
