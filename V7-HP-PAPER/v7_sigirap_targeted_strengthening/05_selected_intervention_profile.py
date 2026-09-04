#!/usr/bin/env python3
"""Inference-safe profile of frozen selected interventions."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import joblib
import numpy as np

from sigirap_common import FIGURES, OUTPUTS, REPORTS, SPLITS, ensure_layout, iter_jsonl


OUT = OUTPUTS / "intervention_profile"
EPS = 1e-12


def bh_adjust(pairs: list[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(pairs, key=lambda pair: pair[1])
    adjusted: dict[int, float] = {}
    previous = 1.0
    count = len(ordered)
    for reverse_rank, (index, value) in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        current = min(previous, value * count / rank)
        adjusted[index] = min(1.0, current)
        previous = current
    return adjusted


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def official_pair_rows(split: str) -> dict[str, dict[str, dict[str, Any]]]:
    if split == "holdout3000":
        path = SPLITS[split]["actions"].parent / "official_metrics/flan_per_query.jsonl"
        names = {"baseline": "baseline", "v4_selected": "full"}
    else:
        path = SPLITS[split]["actions"].parent / "official_per_query_3405.jsonl"
        names = {"frozen_top5_baseline": "baseline", "full_v4": "full"}
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(path):
        if str(row["method"]) in names:
            result[str(row["query_id"])][names[str(row["method"])]] = row
    return result


def action_features(action: dict[str, Any], query: dict[str, Any], selection: dict[str, Any], cache: dict[str, Any]) -> dict[str, float]:
    details = query["doc_feature_details"]
    baseline = [details[doc_id] for doc_id in query["baseline_ids"]]
    candidate = [details[doc_id] for doc_id in query["candidate_ids"]]
    safe = action.get("inference_safe_features", {})
    missing = safe.get("missing_hop_probabilities", {})
    pair_index = cache["pair_feature_names"].index("semantic_complementarity")
    pair_values = [float(values[pair_index]) for values in query["pair_features"].values()]
    context_text = " ".join(str(doc["text"]) for doc in action["context_docs"])[:3200]
    original = list(query["baseline_ids"])
    current = list(action["context_doc_ids"])
    retained = [doc_id for doc_id in current if doc_id in original]
    displacement = mean(abs(current.index(doc_id) - original.index(doc_id)) for doc_id in retained) if retained else 5.0
    return {
        "baseline_cross_encoder_mean": mean(float(row["cross_encoder_relevance"]) for row in baseline),
        "baseline_bm25_mean": mean(float(row["bm25"]) for row in baseline),
        "baseline_entity_overlap_mean": mean(float(row["entity_overlap"]) for row in baseline),
        "candidate_cross_encoder_max": max(float(row["cross_encoder_relevance"]) for row in candidate),
        "candidate_entity_overlap_max": max(float(row["entity_overlap"]) for row in candidate),
        "candidate_bridge_match_max": max(float(row["bridge_entity_match"]) for row in candidate),
        "pair_complementarity_mean": mean(pair_values) if pair_values else 0.0,
        "generator_score": float(action.get("generator_score", 0.0)),
        "added_doc_opportunity_mean": float(safe.get("added_doc_opportunity_mean", 0.0)),
        "added_doc_semantic_mean": float(safe.get("added_doc_semantic_mean", 0.0)),
        "removal_risk": float(safe.get("removal_risk", 0.0)),
        "missing_answer_resolution": float(missing.get("missing_answer_resolution", 0.0)),
        "missing_bridge": float(missing.get("missing_bridge", 0.0)),
        "no_intervention_needed": float(missing.get("no_intervention_needed", 0.0)),
        "ordering_problem": float(missing.get("ordering_problem", 0.0)),
        "redundant_context": float(missing.get("redundant_context", 0.0)),
        "position_displacement": displacement,
        "documents_added": float(len(action.get("added_doc_ids", []))),
        "documents_removed": float(len(action.get("removed_doc_ids", []))),
        "candidate_pool_size": float(len(query["docs"])),
        "context_token_length": float(len(context_text.split())),
        "preservation_probability": float(selection.get("pred_answer_safe_prob", 0.0)),
        "utility_probability": float(selection.get("pred_positive_prob", 0.0)),
    }


def main() -> None:
    ensure_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in ("holdout3000", "revision3405"):
        selections = {
            str(row["query_id"]): row
            for row in iter_jsonl(SPLITS[split]["selections"])
            if bool(row["selected"])
        }
        selected_ids = {str(row["action_id"]) for row in selections.values()}
        actions = {
            str(row["action_id"]): row
            for row in iter_jsonl(SPLITS[split]["actions"])
            if str(row["action_id"]) in selected_ids
        }
        official = official_pair_rows(split)
        cache = joblib.load(SPLITS[split]["cache"])
        for query_id, selection in selections.items():
            baseline = official[query_id]["baseline"]
            full = official[query_id]["full"]
            answer_delta = float(full["answer_f1"]) - float(baseline["answer_f1"])
            joint_delta = float(full["joint_f1"]) - float(baseline["joint_f1"])
            action = actions[str(selection["action_id"])]
            rows.append({
                "split": split,
                "query_id": query_id,
                "action_family": str(action["action_family"]),
                "answer_delta": answer_delta,
                "joint_delta": joint_delta,
                "answer_outcome": "gain" if answer_delta > EPS else "loss" if answer_delta < -EPS else "tie",
                "joint_outcome": "gain" if joint_delta > EPS else "loss" if joint_delta < -EPS else "tie",
                **action_features(action, cache["queries"][query_id], selection, cache),
            })

    feature_names = [
        key for key in rows[0]
        if key not in {"split", "query_id", "action_family", "answer_delta", "joint_delta", "answer_outcome", "joint_outcome"}
    ]
    from scipy.stats import spearmanr
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    analysis_rows: list[dict[str, Any]] = []
    for outcome_name, delta_name in (("answer_gain", "answer_delta"), ("joint_gain", "joint_delta")):
        y = np.asarray([float(row[delta_name]) > EPS for row in rows], dtype=int)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=20260715),
        )
        model.fit(x, y)
        coefficients = model[-1].coef_[0]
        for index, feature in enumerate(feature_names):
            values = [float(row[feature]) for row in rows]
            correlation = spearmanr(values, [float(row[delta_name]) for row in rows])
            threshold = float(np.quantile(values, 0.75))
            high = [row for row in rows if float(row[feature]) >= threshold]
            low = [row for row in rows if float(row[feature]) < threshold]
            analysis_rows.append({
                "outcome": outcome_name,
                "feature": feature,
                "n": len(rows),
                "spearman_rho": float(correlation.statistic) if not math.isnan(float(correlation.statistic)) else 0.0,
                "spearman_p": float(correlation.pvalue) if not math.isnan(float(correlation.pvalue)) else 1.0,
                "standardized_logistic_coefficient": float(coefficients[index]),
                "odds_ratio_per_sd": math.exp(float(coefficients[index])),
                "top_quartile_gain_rate": mean(float(row[delta_name]) > EPS for row in high),
                "lower_75_gain_rate": mean(float(row[delta_name]) > EPS for row in low) if low else 0.0,
                "univariate_gain_rate_difference": mean(float(row[delta_name]) > EPS for row in high) - (mean(float(row[delta_name]) > EPS for row in low) if low else 0.0),
            })
    adjustments = bh_adjust([(index, float(row["spearman_p"])) for index, row in enumerate(analysis_rows)])
    for index, row in enumerate(analysis_rows):
        row["spearman_p_bh"] = adjustments[index]
    write_csv(OUT / "feature_analysis.csv", analysis_rows)

    family_rows = []
    for family in sorted({row["action_family"] for row in rows}):
        values = [row for row in rows if row["action_family"] == family]
        family_rows.append({
            "action_family": family,
            "n": len(values),
            "answer_gain_rate": mean(row["answer_outcome"] == "gain" for row in values),
            "answer_tie_rate": mean(row["answer_outcome"] == "tie" for row in values),
            "answer_loss_rate": mean(row["answer_outcome"] == "loss" for row in values),
            "joint_gain_rate": mean(row["joint_outcome"] == "gain" for row in values),
            "joint_tie_rate": mean(row["joint_outcome"] == "tie" for row in values),
            "joint_loss_rate": mean(row["joint_outcome"] == "loss" for row in values),
            "mean_answer_delta": mean(row["answer_delta"] for row in values),
            "mean_joint_delta": mean(row["joint_delta"] for row in values),
        })
    write_csv(OUT / "action_family_success.csv", family_rows)

    calibration_rows = []
    for score_name, observed_name in (("preservation_probability", "answer_outcome"), ("utility_probability", "joint_outcome")):
        for bin_index in range(5):
            low, high = bin_index / 5, (bin_index + 1) / 5
            members = [
                row for row in rows
                if low <= float(row[score_name]) < high or (bin_index == 4 and float(row[score_name]) == 1.0)
            ]
            if members:
                calibration_rows.append({
                    "score": score_name,
                    "bin_low": low,
                    "bin_high": high,
                    "n": len(members),
                    "mean_predicted_score": mean(float(row[score_name]) for row in members),
                    "observed_success_rate": mean(row[observed_name] != "loss" for row in members) if score_name == "preservation_probability" else mean(row[observed_name] == "gain" for row in members),
                })
    write_csv(OUT / "calibration_bins.csv", calibration_rows)

    import matplotlib.pyplot as plt

    names = [row["action_family"].replace("_", "\n") for row in family_rows]
    answer = [row["answer_gain_rate"] for row in family_rows]
    joint = [row["joint_gain_rate"] for row in family_rows]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(positions - 0.18, answer, width=0.36, label="Answer gain", color="#2a7f62")
    ax.bar(positions + 0.18, joint, width=0.36, label="Joint gain", color="#235789")
    ax.set_xticks(positions, names, fontsize=7)
    ax.set_ylabel("Selected-intervention gain rate")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "intervention_success_by_action.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for score_name in ("preservation_probability", "utility_probability"):
        values = [row for row in calibration_rows if row["score"] == score_name]
        ax.plot([row["mean_predicted_score"] for row in values], [row["observed_success_rate"] for row in values], marker="o", label=score_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed target rate")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "preservation_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    strongest_answer = sorted([row for row in analysis_rows if row["outcome"] == "answer_gain"], key=lambda row: abs(row["standardized_logistic_coefficient"]), reverse=True)[:5]
    strongest_joint = sorted([row for row in analysis_rows if row["outcome"] == "joint_gain"], key=lambda row: abs(row["standardized_logistic_coefficient"]), reverse=True)[:5]
    report = [
        "# Selected Intervention Success/Failure Profile",
        "",
        "## Scope",
        "",
        f"The analysis covers {len(rows)} interventions selected by the already frozen policy on the two Hotpot holdouts. All predictors are available before the final reader call: retrieval scores, entity/bridge overlap, pair-complementarity proxies, bounded action structure, context length, and frozen preservation/utility probabilities. Gold answer/support features, post-reader outcomes, and oracle utility are excluded as predictors. Results are explanatory only and do not retrain the selector or alter thresholds.",
        "",
        "## Strongest standardized associations",
        "",
        "| Outcome | Feature | Logistic coefficient | OR per SD | Spearman rho | BH-FDR p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in strongest_answer + strongest_joint:
        report.append(
            f"| {row['outcome']} | {row['feature']} | {row['standardized_logistic_coefficient']:+.3f} | "
            f"{row['odds_ratio_per_sd']:.3f} | {row['spearman_rho']:+.3f} | {row['spearman_p_bh']:.4f} |"
        )
    report.extend([
        "",
        "## Interpretation boundary",
        "",
        "These coefficients describe associations within the policy-selected subset. Selection changes the feature distribution, correlated predictors make coefficients non-causal, and multiple comparisons are controlled only for the Spearman family. The profile is not used to propose a new selector in this submission cycle.",
    ])
    (REPORTS / "intervention_profile_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "n_selected": len(rows), "action_families": Counter(row["action_family"] for row in rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
