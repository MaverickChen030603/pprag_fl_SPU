#!/usr/bin/env python3
"""Run answer, sentence-support, and joint HotpotQA metrics on frozen v4 selections."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v4_common import (
    OUTPUTS, REPORTS, answer_scores, capitalized_entities, ensure_layout, jaccard,
    normalize_answer, normalize_title, overlap_ratio, read_json, read_jsonl, tokens,
    write_json, write_jsonl,
)


DEFAULT_ARROW = "/home/iiserver31/.cache/huggingface/datasets/hotpot_qa/distractor/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab/hotpot_qa-validation.arrow"
SEED = 20260714


def load_official(path: str) -> dict[str, dict[str, Any]]:
    from datasets import Dataset

    return {str(row["id"]): row for row in Dataset.from_file(path)}


def sentence_features(question: str, title: str, sentence: str, doc_rank: int, sent_id: int, count: int) -> list[float]:
    question_tokens, sentence_tokens, title_tokens = tokens(question), tokens(sentence), tokens(title)
    return [
        overlap_ratio(question_tokens, sentence_tokens),
        overlap_ratio(question_tokens, title_tokens),
        jaccard(question_tokens, sentence_tokens),
        jaccard(capitalized_entities(question), capitalized_entities(f"{title} {sentence}")),
        float(sent_id == 0), sent_id / max(1, count - 1), doc_rank / 4.0,
        min(len(sentence_tokens), 80) / 80.0, overlap_ratio(title_tokens, sentence_tokens),
    ]


def context_instances(query_id: str, action: dict[str, Any], official: dict[str, Any]) -> list[dict[str, Any]]:
    by_title = {
        normalize_title(title): {"title": title, "sentences": sentences}
        for title, sentences in zip(official["context"]["title"], official["context"]["sentences"])
    }
    gold = {
        (normalize_title(title), int(sentence_id))
        for title, sentence_id in zip(official["supporting_facts"]["title"], official["supporting_facts"]["sent_id"])
    }
    rows = []
    for doc_rank, title in enumerate(action["context_titles"]):
        doc = by_title.get(normalize_title(title))
        if not doc:
            continue
        for sentence_id, sentence in enumerate(doc["sentences"]):
            rows.append({
                "query_id": query_id,
                "title": doc["title"],
                "sent_id": sentence_id,
                "features": sentence_features(official["question"], doc["title"], sentence, doc_rank, sentence_id, len(doc["sentences"])),
                "label": int((normalize_title(doc["title"]), sentence_id) in gold),
            })
    return rows


def fit(rows: list[dict[str, Any]]) -> Any:
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([row["label"] for row in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=SEED))
    model.fit(x, y)
    return model


def score(model: Any, rows: list[dict[str, Any]]) -> list[float]:
    values = model.predict_proba(np.asarray([row["features"] for row in rows], dtype=np.float64))
    classes = list(model.classes_)
    return [float(value[classes.index(1)]) if 1 in classes else 0.0 for value in values]


def support_set(rows: list[dict[str, Any]], scores: list[float], threshold: float) -> set[tuple[str, int]]:
    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)
    selected = [row for row, value in ranked if value >= threshold][:5]
    if len(selected) < 2:
        selected = [row for row, _ in ranked[:2]]
    return {(normalize_title(row["title"]), int(row["sent_id"])) for row in selected}


def support_metrics(prediction: set[tuple[str, int]], gold: set[tuple[str, int]]) -> dict[str, float]:
    true_positive = len(prediction & gold)
    precision = true_positive / len(prediction) if prediction else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": float(prediction == gold), "precision": precision, "recall": recall, "f1": f1}


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    exact_match, f1 = answer_scores(prediction, gold)
    pred, truth = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = Counter(pred) & Counter(truth)
    overlap = sum(common.values())
    return {
        "em": exact_match,
        "f1": f1,
        "precision": overlap / len(pred) if pred else float(pred == truth),
        "recall": overlap / len(truth) if truth else float(pred == truth),
    }


def official_metrics(prediction: str, gold_answer: str, pred_support: set[tuple[str, int]], gold_support: set[tuple[str, int]]) -> dict[str, float]:
    answer = answer_metrics(prediction, gold_answer)
    support = support_metrics(pred_support, gold_support)
    joint_precision = answer["precision"] * support["precision"]
    joint_recall = answer["recall"] * support["recall"]
    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall) if joint_precision + joint_recall else 0.0
    return {
        "answer_em": answer["em"], "answer_f1": answer["f1"],
        "answer_precision": answer["precision"], "answer_recall": answer["recall"],
        "sp_em": support["em"], "sp_f1": support["f1"],
        "sp_precision": support["precision"], "sp_recall": support["recall"],
        "joint_em": answer["em"] * support["em"], "joint_f1": joint_f1,
    }


def paired_bootstrap(differences: list[float], rounds: int = 5000) -> dict[str, float]:
    rng = random.Random(SEED)
    n = len(differences)
    samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds)]
    samples.sort()
    return {
        "mean": mean(differences), "ci95_low": samples[int(0.025 * rounds)], "ci95_high": samples[int(0.975 * rounds)],
        "p_value": min(1.0, 2 * min(sum(value <= 0 for value in samples) / rounds, sum(value >= 0 for value in samples) / rounds)),
    }


def main() -> None:
    ensure_layout()
    nested = read_json(OUTPUTS / "nested_selector/v4_nested_summary.json")
    if nested.get("status") != "complete":
        reason = "Official sentence-level evaluation is gated on a completed frozen nested selector."
        write_json(OUTPUTS / "official_metrics/official_hotpotqa_summary.json", {"status": "skipped_by_opportunity_gate", "reason": reason})
        (REPORTS / "official_metric_report.md").write_text(f"# Official HotpotQA Metrics\n\nStatus: skipped. {reason} Title metrics are not renamed as official metrics.\n", encoding="utf-8")
        print(json.dumps({"status": "skipped_by_opportunity_gate"}, indent=2))
        return
    arrow_path = os.environ.get("V4_HOTPOT_ARROW", DEFAULT_ARROW)
    if not Path(arrow_path).exists():
        raise FileNotFoundError(arrow_path)
    official = load_official(arrow_path)
    selections = read_jsonl(OUTPUTS / "nested_selector/v4_nested_per_query.jsonl")
    actions = {str(row["action_id"]): row for row in read_jsonl(OUTPUTS / "generated_actions/v4_outer_test_actions.jsonl")}
    outcomes = {str(row["action_id"]): row for row in read_jsonl(OUTPUTS / "action_outcomes/v4_action_outputs.jsonl")}
    query_ids = {str(row["query_id"]) for row in selections}
    instances: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for selection in selections:
        query_id = str(selection["query_id"])
        for method, action_id in [("baseline", f"{query_id}::v4::fallback"), ("v4_selected", str(selection["action_id"]))]:
            instances[(query_id, method)] = context_instances(query_id, actions[action_id], official[query_id])

    fold_thresholds, fold_models = {}, {}
    for outer_fold in range(5):
        test_ids = {str(row["query_id"]) for row in selections if int(row["outer_fold"]) == outer_fold}
        train_ids = query_ids - test_ids
        train_rows = [row for query_id in train_ids for method in ("baseline", "v4_selected") for row in instances[(query_id, method)]]
        inner_ids = [set(sorted(train_ids, key=lambda value: int(hashlib.md5(f"sp-{outer_fold}-{value}".encode()).hexdigest(), 16))[index::3]) for index in range(3)]
        oof: dict[tuple[str, str, int], float] = {}
        for validation_ids in inner_ids:
            fit_rows = [row for row in train_rows if row["query_id"] not in validation_ids]
            validation_rows = [row for row in train_rows if row["query_id"] in validation_ids]
            model = fit(fit_rows)
            for row, value in zip(validation_rows, score(model, validation_rows)):
                oof[(row["query_id"], normalize_title(row["title"]), int(row["sent_id"]))] = value
        candidates = []
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            values = []
            for query_id in train_ids:
                gold = {(normalize_title(title), int(sentence_id)) for title, sentence_id in zip(official[query_id]["supporting_facts"]["title"], official[query_id]["supporting_facts"]["sent_id"])}
                for method in ("baseline", "v4_selected"):
                    rows = instances[(query_id, method)]
                    values.append(support_metrics(support_set(rows, [oof[(row["query_id"], normalize_title(row["title"]), int(row["sent_id"]))] for row in rows], threshold), gold)["f1"])
            candidates.append((mean(values), threshold))
        fold_thresholds[outer_fold] = max(candidates)[1]
        fold_models[outer_fold] = fit(train_rows)

    metric_rows = []
    for selection in selections:
        query_id, outer_fold = str(selection["query_id"]), int(selection["outer_fold"])
        gold_support = {(normalize_title(title), int(sentence_id)) for title, sentence_id in zip(official[query_id]["supporting_facts"]["title"], official[query_id]["supporting_facts"]["sent_id"])}
        for method, action_id in [("baseline", f"{query_id}::v4::fallback"), ("v4_selected", str(selection["action_id"]))]:
            rows = instances[(query_id, method)]
            prediction_support = support_set(rows, score(fold_models[outer_fold], rows), fold_thresholds[outer_fold])
            metrics = official_metrics(outcomes[action_id]["prediction"], official[query_id]["answer"], prediction_support, gold_support)
            metric_rows.append({"query_id": query_id, "method": method, "outer_fold": outer_fold, **metrics})
    write_jsonl(OUTPUTS / "official_metrics/official_hotpotqa_per_query.jsonl", metric_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["method"]].append(row)
    metric_names = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1"]
    summary = {method: {metric: mean(row[metric] for row in rows) for metric in metric_names} for method, rows in grouped.items()}
    by_query = defaultdict(dict)
    for row in metric_rows:
        by_query[row["query_id"]][row["method"]] = row
    significance = {
        metric: paired_bootstrap([values["v4_selected"][metric] - values["baseline"][metric] for values in by_query.values()])
        for metric in metric_names
    }
    payload = {"status": "complete", "n_queries": len(query_ids), "metrics": summary, "significance": significance, "fold_thresholds": fold_thresholds, "title_proxy_renamed_as_official": False}
    write_json(OUTPUTS / "official_metrics/official_hotpotqa_summary.json", payload)
    write_json(OUTPUTS / "official_metrics/official_hotpotqa_significance.json", significance)
    (REPORTS / "official_metric_report.md").write_text(
        f"# Official HotpotQA Metrics\n\nCompleted on {len(query_ids)} frozen outer-test queries with a nested sentence-support predictor. Joint F1 delta: **{significance['joint_f1']['mean']:+.4f}** (p={significance['joint_f1']['p_value']:.4f}).\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
