#!/usr/bin/env python3
"""Official sentence-level HotpotQA evaluation with a nested support predictor."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v3_common import OUTPUTS, REPORTS, answer_scores, capitalized_entities, ensure_layout, jaccard, markdown_table, normalize_answer, normalize_title, overlap_ratio, paired_bootstrap, read_json, read_jsonl, tokens, write_json, write_jsonl


DEFAULT_ARROW = "/home/iiserver31/.cache/huggingface/datasets/hotpot_qa/distractor/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab/hotpot_qa-validation.arrow"
SEED = 20260713


def qhash(query_id: str, salt: str) -> int:
    return int(hashlib.md5((salt + query_id).encode("utf-8")).hexdigest(), 16)


def split_ids(ids: set[str], folds: int, salt: str) -> list[set[str]]:
    ordered = sorted(ids, key=lambda value: qhash(value, salt))
    return [set(ordered[index::folds]) for index in range(folds)]


def load_official(path: str) -> dict[str, dict[str, Any]]:
    from datasets import Dataset

    dataset = Dataset.from_file(path)
    return {str(row["id"]): row for row in dataset}


def action_docs() -> dict[str, dict[str, Any]]:
    return {str(row["action_id"]): row for row in read_jsonl(OUTPUTS / "candidate_generation/v3_candidate_actions.jsonl")}


def text_similarity(left: str, right: str) -> float:
    return jaccard(tokens(left), tokens(right))


def map_context(action: dict[str, Any], example: dict[str, Any]) -> list[dict[str, Any]]:
    official_docs = [
        {"title": title, "sentences": sentences}
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"])
    ]
    by_title = {normalize_title(doc["title"]): doc for doc in official_docs}
    mapped: list[dict[str, Any]] = []
    used: set[str] = set()
    for rank, source_doc in enumerate(action["context_docs"]):
        doc = by_title.get(normalize_title(source_doc["title"]))
        if doc is None:
            candidates = [value for value in official_docs if normalize_title(value["title"]) not in used]
            doc = max(candidates, key=lambda value: text_similarity(source_doc.get("text", ""), " ".join(value["sentences"])))
        used.add(normalize_title(doc["title"]))
        mapped.append({"title": doc["title"], "sentences": doc["sentences"], "doc_rank": rank})
    return mapped


def sentence_features(question: str, title: str, sentence: str, doc_rank: int, sent_id: int, num_sentences: int) -> list[float]:
    q_tokens, s_tokens, title_tokens = tokens(question), tokens(sentence), tokens(title)
    q_entities = capitalized_entities(question)
    sentence_entities = capitalized_entities(f"{title} {sentence}")
    return [
        overlap_ratio(q_tokens, s_tokens),
        overlap_ratio(q_tokens, title_tokens),
        jaccard(q_tokens, s_tokens),
        jaccard(q_entities, sentence_entities),
        float(sent_id == 0),
        sent_id / max(1, num_sentences - 1),
        doc_rank / 4.0,
        min(len(s_tokens), 80) / 80.0,
        overlap_ratio(title_tokens, s_tokens),
    ]


def build_instances(selector_rows: list[dict[str, Any]], actions: dict[str, dict[str, Any]], official: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    unique: dict[tuple[str, str, int], dict[str, Any]] = {}
    instances: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for selection in selector_rows:
        query_id = str(selection["query_id"])
        example = official[query_id]
        gold = {(normalize_title(title), int(sent_id)) for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"])}
        for method, action_id in [("baseline", f"{query_id}::fallback"), ("v3_selected", str(selection["action_id"]))]:
            action = actions[action_id]
            rows: list[dict[str, Any]] = []
            for doc in map_context(action, example):
                for sent_id, sentence in enumerate(doc["sentences"]):
                    key = (query_id, normalize_title(doc["title"]), sent_id)
                    row = {
                        "query_id": query_id,
                        "title": doc["title"],
                        "sent_id": sent_id,
                        "sentence": sentence,
                        "features": sentence_features(example["question"], doc["title"], sentence, doc["doc_rank"], sent_id, len(doc["sentences"])),
                        "label": int((normalize_title(doc["title"]), sent_id) in gold),
                    }
                    unique.setdefault(key, row)
                    rows.append(row)
            instances[(query_id, method)] = rows
    return list(unique.values()), instances


def fit_model(rows: list[dict[str, Any]]):
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=1500, random_state=SEED))
    model.fit(x, y)
    return model


def predict(model: Any, rows: list[dict[str, Any]]) -> list[float]:
    values = model.predict_proba(np.asarray([row["features"] for row in rows], dtype=np.float64))
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0] * len(rows)
    return [float(value[classes.index(1)]) for value in values]


def predicted_support(rows: list[dict[str, Any]], score_map: dict[tuple[str, str, int], float], threshold: float) -> set[tuple[str, int]]:
    ranked = sorted(rows, key=lambda row: score_map[(row["query_id"], normalize_title(row["title"]), int(row["sent_id"]))], reverse=True)
    selected = [row for row in ranked if score_map[(row["query_id"], normalize_title(row["title"]), int(row["sent_id"]))] >= threshold][:5]
    if len(selected) < 2:
        selected = ranked[:2]
    return {(normalize_title(row["title"]), int(row["sent_id"])) for row in selected}


def sp_scores(pred: set[tuple[str, int]], gold: set[tuple[str, int]]) -> dict[str, float]:
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": float(pred == gold), "precision": precision, "recall": recall, "f1": f1}


def answer_prf(prediction: str, gold: str) -> dict[str, float]:
    pred, truth = normalize_answer(prediction).split(), normalize_answer(gold).split()
    em, f1 = answer_scores(prediction, gold)
    if not pred or not truth:
        precision = recall = float(pred == truth)
    else:
        common = Counter(pred) & Counter(truth)
        same = sum(common.values())
        precision = same / len(pred)
        recall = same / len(truth)
    return {"em": em, "precision": precision, "recall": recall, "f1": f1}


def official_row(query_id: str, method: str, prediction: str, gold_answer: str, pred_support: set[tuple[str, int]], gold_support: set[tuple[str, int]]) -> dict[str, Any]:
    answer = answer_prf(prediction, gold_answer)
    support = sp_scores(pred_support, gold_support)
    joint_precision = answer["precision"] * support["precision"]
    joint_recall = answer["recall"] * support["recall"]
    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall) if joint_precision + joint_recall else 0.0
    return {
        "query_id": query_id,
        "method": method,
        "prediction": prediction,
        "predicted_supporting_facts": sorted([list(value) for value in pred_support]),
        "answer_em": answer["em"],
        "answer_f1": answer["f1"],
        "sp_em": support["em"],
        "sp_f1": support["f1"],
        "sp_precision": support["precision"],
        "sp_recall": support["recall"],
        "joint_em": answer["em"] * support["em"],
        "joint_f1": joint_f1,
    }


def main() -> None:
    ensure_layout()
    nested_path = OUTPUTS / "nested_selector/v3_nested_summary.json"
    nested = read_json(nested_path) if nested_path.exists() else {"status": "not_run"}
    if nested.get("status") != "complete":
        reason = "Official sentence-support evaluation requires frozen v3 outer-test contexts, which were not produced after the opportunity gate failed."
        write_json(OUTPUTS / "official_metrics/official_hotpotqa_summary.json", {"status": "skipped_by_opportunity_gate", "reason": reason})
        write_json(OUTPUTS / "official_metrics/official_hotpotqa_significance.json", {"status": "not_run", "reason": reason})
        (OUTPUTS / "tables/official_hotpotqa_main_table.md").write_text("# Official HotpotQA Metrics\n\nStatus: **skipped by opportunity gate**. No title proxy is renamed as an official sentence metric.\n", encoding="utf-8")
        (REPORTS / "official_metric_report.md").write_text("# Official HotpotQA Metric Report\n\nStatus: **skipped by opportunity gate**. " + reason + "\n", encoding="utf-8")
        print(json.dumps({"status": "skipped_by_opportunity_gate"}, indent=2))
        return
    arrow = os.environ.get("V3_HOTPOT_ARROW", DEFAULT_ARROW)
    if not Path(arrow).exists():
        write_json(OUTPUTS / "official_metrics/official_hotpotqa_summary.json", {"status": "blocked_missing_official_arrow", "path": arrow})
        raise FileNotFoundError(arrow)
    official = load_official(arrow)
    selections = read_jsonl(OUTPUTS / "nested_selector/v3_nested_per_query.jsonl")
    action_map = action_docs()
    unique_rows, instances = build_instances(selections, action_map, official)
    query_ids = {str(row["query_id"]) for row in selections}
    outer_sets = [{row["query_id"] for row in selections if int(row["outer_fold"]) == fold} for fold in range(5)]
    score_map: dict[tuple[str, str, int], float] = {}
    fold_configs: list[dict[str, Any]] = []

    for outer_fold, test_ids in enumerate(outer_sets):
        train_ids = query_ids - test_ids
        train_rows = [row for row in unique_rows if row["query_id"] in train_ids]
        test_rows = [row for row in unique_rows if row["query_id"] in test_ids]
        inner_sets = split_ids(train_ids, 5, f"official-support-{outer_fold}")
        oof_scores: dict[tuple[str, str, int], float] = {}
        for validation_ids in inner_sets:
            fit_rows = [row for row in train_rows if row["query_id"] not in validation_ids]
            validation_rows = [row for row in train_rows if row["query_id"] in validation_ids]
            model = fit_model(fit_rows)
            for row, score in zip(validation_rows, predict(model, validation_rows)):
                oof_scores[(row["query_id"], normalize_title(row["title"]), int(row["sent_id"]))] = score
        threshold_scores: list[tuple[float, float]] = []
        for threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            values: list[float] = []
            for query_id in train_ids:
                gold = {(normalize_title(title), int(sent_id)) for title, sent_id in zip(official[query_id]["supporting_facts"]["title"], official[query_id]["supporting_facts"]["sent_id"])}
                for method in ("baseline", "v3_selected"):
                    pred = predicted_support(instances[(query_id, method)], oof_scores, threshold)
                    values.append(sp_scores(pred, gold)["f1"])
            threshold_scores.append((mean(values), threshold))
        _, threshold = max(threshold_scores)
        model = fit_model(train_rows)
        for row, score in zip(test_rows, predict(model, test_rows)):
            score_map[(row["query_id"], normalize_title(row["title"]), int(row["sent_id"]))] = score
        fold_configs.append({
            "outer_fold": outer_fold,
            "n_train_queries": len(train_ids),
            "n_test_queries": len(test_ids),
            "train_selected_threshold": threshold,
            "inner_oof_threshold_scores": [{"threshold": value, "mean_sp_f1": score} for score, value in threshold_scores],
            "outer_test_labels_used_for_training_or_threshold": False,
        })

    flan_rows = read_jsonl(OUTPUTS / "action_outcomes/v3_action_reader_outputs.jsonl")
    flan_by_action = {str(row["action_id"]): row for row in flan_rows}
    selection_by_query = {str(row["query_id"]): row for row in selections}
    threshold_by_fold = {row["outer_fold"]: row["train_selected_threshold"] for row in fold_configs}
    metric_rows: list[dict[str, Any]] = []
    for query_id in sorted(query_ids):
        selection = selection_by_query[query_id]
        threshold = threshold_by_fold[int(selection["outer_fold"])]
        example = official[query_id]
        gold_support = {(normalize_title(title), int(sent_id)) for title, sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"])}
        for method, action_id in [("baseline", f"{query_id}::fallback"), ("v3_selected", str(selection["action_id"]))]:
            pred_support = predicted_support(instances[(query_id, method)], score_map, threshold)
            metric_rows.append(official_row(query_id, method, flan_by_action[action_id]["prediction"], example["answer"], pred_support, gold_support))
    write_jsonl(OUTPUTS / "official_metrics/official_hotpotqa_per_query.jsonl", metric_rows)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_method[row["method"]].append(row)
    metrics = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1"]
    summary = {method: {metric: mean(row[metric] for row in rows) for metric in metrics} | {"n": len(rows)} for method, rows in by_method.items()}
    summary["status"] = "complete"
    summary["support_predictor"] = "five-outer-fold nested sentence classifier with inner-OOF threshold selection"
    write_json(OUTPUTS / "official_metrics/official_hotpotqa_summary.json", summary)
    paired: dict[str, list[float]] = {metric: [] for metric in metrics}
    baseline_map = {row["query_id"]: row for row in by_method["baseline"]}
    selected_map = {row["query_id"]: row for row in by_method["v3_selected"]}
    for query_id in query_ids:
        for metric in metrics:
            paired[metric].append(float(selected_map[query_id][metric]) - float(baseline_map[query_id][metric]))
    significance = {metric: paired_bootstrap(values, seed=SEED + index) for index, (metric, values) in enumerate(paired.items())}
    significance["protocol"] = {"official_sentence_ids": True, "title_proxy_renamed": False, "nested_support_predictor": True, "fold_configs": fold_configs}
    write_json(OUTPUTS / "official_metrics/official_hotpotqa_significance.json", significance)
    table_rows = [[method, *[f"{summary[method][metric]:.4f}" for metric in metrics]] for method in ["baseline", "v3_selected"]]
    table_rows.append(["delta", *[f"{significance[metric]['mean']:+.4f}" for metric in metrics]])
    table = markdown_table(["Method", "Answer EM", "Answer F1", "SP EM", "SP F1", "Joint EM", "Joint F1"], table_rows)
    (OUTPUTS / "tables/official_hotpotqa_main_table.md").write_text("# Official HotpotQA Metrics\n\n" + table + "\n", encoding="utf-8")
    report = f"""# Official HotpotQA Metric Report

This evaluation predicts sentence-level supporting-fact IDs with an outer-fold nested classifier. Its threshold is selected only from inner-OOF outer-train predictions. The reported `sp_f1` and `joint_f1` are official-style sentence metrics, not renamed title proxies.

{table}

Paired bootstrap: SP F1 delta {significance['sp_f1']['mean']:+.4f}, p={significance['sp_f1']['p_value']:.4f}; Joint F1 delta {significance['joint_f1']['mean']:+.4f}, p={significance['joint_f1']['p_value']:.4f}.
"""
    (REPORTS / "official_metric_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": summary, "joint_f1_significance": significance["joint_f1"], "sp_f1_significance": significance["sp_f1"]}, indent=2))


if __name__ == "__main__":
    main()
