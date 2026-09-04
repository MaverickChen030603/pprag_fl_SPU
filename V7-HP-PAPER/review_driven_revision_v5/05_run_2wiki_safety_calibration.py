#!/usr/bin/env python3
"""Few-shot 2Wiki safety calibration with a frozen generator and reader."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from v5_common import HERE, V4, V4_COMPLETION, config, iter_jsonl, paired_bootstrap, read_json, read_jsonl, write_json, write_jsonl, write_text


OUT = HERE / "outputs" / "2wiki_calibration"
EXTERNAL = V4_COMPLETION / "outputs" / "external_2wiki_frozen"
RAW_TRAIN = Path("/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/cross_dataset_validation/raw/2wiki/extracted/data/train.json")
FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
METRICS = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")


def load_module(path: Path, name: str) -> Any:
    for module_path in (path.parent, V4, V4_COMPLETION, V4.parent.parent):
        if str(module_path) not in sys.path:
            sys.path.insert(0, str(module_path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_raw(item: dict[str, Any], index: int) -> dict[str, Any]:
    query_id = str(item["_id"])
    return {
        "query_id": query_id,
        "question": str(item["question"]),
        "answer": str(item["answer"]),
        "supporting_facts": list(item.get("supporting_facts", [])),
        "supporting_titles": sorted({str(value[0]) for value in item.get("supporting_facts", [])}),
        "context": list(item["context"]),
        "type": str(item.get("type", "")),
        "source_dataset": "2WikiMultiHopQA",
        "split": "train_calibration",
        "sample_index": index,
    }


def text_doc(query_id: str, index: int, raw: list[Any]) -> dict[str, Any]:
    title, sentences = str(raw[0]), [str(value) for value in raw[1]]
    return {"doc_id": f"{query_id}::doc_{index}", "title": title, "text": " ".join(sentences), "sentences": sentences, "source_rank": index}


def calibration_ids(raw: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    cfg = config()["two_wiki_calibration"]
    by_id = {str(row["_id"]): row for row in raw}
    sets: dict[str, dict[str, list[str]]] = {}
    union = set()
    for seed in cfg["seeds"]:
        ranked = sorted(by_id, key=lambda query_id: hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest())
        sets[str(seed)] = {}
        for k in cfg["k_shots"]:
            ids = ranked[: int(k)]
            sets[str(seed)][str(k)] = ids
            union.update(ids)
    return sorted(union), sets


def prepare_stage(args: argparse.Namespace) -> None:
    if not RAW_TRAIN.exists():
        raise FileNotFoundError(RAW_TRAIN)
    raw = json.loads(RAW_TRAIN.read_text(encoding="utf-8"))
    ids, sets = calibration_ids(raw)
    by_id = {str(row["_id"]): row for row in raw}
    sampled = [normalize_raw(by_id[query_id], index) for index, query_id in enumerate(ids)]
    sys.path.insert(0, str(V4.parent.parent))
    from src.v7_hp4.hybrid_retriever import HybridDocument, HybridSoftRetriever

    contexts = []
    for item in sampled:
        docs = [text_doc(item["query_id"], index, value) for index, value in enumerate(item["context"])]
        ranked = HybridSoftRetriever(
            [HybridDocument(doc_id=row["doc_id"], title=row["title"], text=row["text"], soft_weight=1.0) for row in docs],
            alpha=0.55,
        ).rank(item["question"], top_k=min(5, len(docs)))
        by_doc = {row["doc_id"]: row for row in docs}
        baseline = [by_doc[doc.doc_id] for doc, _ in ranked]
        contexts.append(
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "baseline_doc_ids": [row["doc_id"] for row in baseline],
                "baseline_titles": [row["title"] for row in baseline],
                "baseline_context": baseline,
                "all_docs": docs,
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "calibration_source_union.json", sampled)
    write_json(OUT / "calibration_sets.json", sets)
    write_jsonl(OUT / "calibration_contexts_union.jsonl", contexts)
    xfer = load_module(V4_COMPLETION / "02_generate_and_select_2wiki_frozen.py", "v5_2wiki_calibration_xfer")
    source, snapshots = xfer.source_and_snapshots(contexts)
    train_module = load_module(V4 / "03_train_semantic_candidate_generator.py", "v5_2wiki_calibration_features")
    cache_path = OUT / "semantic_feature_cache_union.joblib"
    if args.reuse_cache and cache_path.exists():
        cache = joblib.load(cache_path)
    else:
        cache = train_module.compute_cache(source, snapshots, train_module.DEFAULT_BI_ENCODER, train_module.DEFAULT_CROSS_ENCODER, args.device, args.batch_size)
        joblib.dump(cache, cache_path, compress=3)
    generate = load_module(V4 / "14_generate_frozen_scaleup_actions.py", "v5_2wiki_calibration_generate")
    selector = load_module(V4 / "07_train_nested_selector_v4.py", "v5_2wiki_calibration_selector")
    import v4_common

    manifest = read_json(V4 / "outputs/semantic_generator/foldwise_generator_models.json")
    actions = []
    for fold in v4_common.build_folds(source):
        fold_id = int(fold["fold_id"])
        generator_record = next(row for row in manifest["folds"] if int(row["fold_id"]) == fold_id)
        generator = joblib.load(generator_record["model_path"])
        selector_bundle = joblib.load(V4 / "outputs/scaleup/selector_models" / f"fold_{fold_id}_selector.joblib")
        for query_id in fold["test_query_ids"]:
            generated = generate.generate_actions(query_id, fold_id, cache["queries"][query_id], generator, 8)
            effective = [row for row in generated if row["action_family"] != "fallback"]
            safe = selector.probabilities(selector_bundle["safety_model"], effective)
            positive = selector.probabilities(selector_bundle["opportunity_model"], effective)
            for row, safe_value, positive_value in zip(effective, safe, positive):
                row["pred_answer_safe_prob"] = safe_value
                row["pred_positive_prob"] = positive_value
            for row in generated:
                row["action_id"] = str(row["action_id"]).replace("::v4scale::", "::v5cal::")
                row["calibration_only"] = True
            actions.extend(generated)
    write_jsonl(OUT / "calibration_actions_union.jsonl", actions)
    write_json(
        OUT / "calibration_preparation_audit.json",
        {
            "status": "pass",
            "source_split": "2Wiki train",
            "source_queries": len(raw),
            "union_queries": len(ids),
            "sets": sets,
            "evaluation_query_overlap": len(set(ids) & {str(row["query_id"]) for row in read_json(EXTERNAL / "2wiki_frozen_1000.json")}),
            "generator_frozen": True,
            "selector_heads_frozen": True,
            "reader_frozen": True,
            "evaluation_outcomes_used": False,
            "n_action_rows": len(actions),
        },
    )
    print(json.dumps({"status": "pass", "union_queries": len(ids), "action_rows": len(actions)}, indent=2))


def reader_prompt(question: str, docs: list[dict[str, Any]]) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, 1))
    return "Answer the question using only the context. Return a short answer.\n\n" f"Question: {question}\n\nContext:\n{context[:3200]}\n\nAnswer:"


def reader_stage(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    actions = read_jsonl(OUT / "calibration_actions_union.jsonl")
    assigned = [row for index, row in enumerate(actions) if index % args.num_shards == args.shard_id]
    output = OUT / "reader" / f"outcomes.shard{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output) if args.resume and output.exists() else []
    done = {str(row["action_id"]) for row in existing}
    pending = [row for row in assigned if str(row["action_id"]) not in done]
    source = {str(row["query_id"]): row for row in read_json(OUT / "calibration_source_union.json")}
    sys.path.insert(0, str(V4))
    import v4_common

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float16).to(args.device)
    model.eval()
    rows = list(existing)
    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        encoded = tokenizer([reader_prompt(row["question"], row["context_docs"]) for row in batch], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for action, prediction in zip(batch, predictions):
            item = source[str(action["query_id"])]
            answer_em, answer_f1 = v4_common.answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = v4_common.title_metrics(action["context_titles"], item["supporting_titles"])
            rows.append({"query_id": action["query_id"], "action_id": action["action_id"], "prediction": prediction.strip(), "answer_em": answer_em, "answer_f1": answer_f1, "title_recall": title_recall, "title_f1": title_f1, "answer_title_product": answer_f1 * title_f1})
        if (start // args.batch_size) % 10 == 0 or start + args.batch_size >= len(pending):
            write_jsonl(output, rows)
            write_json(OUT / "reader" / f"progress_shard{args.shard_id}.json", {"status": "running" if start + args.batch_size < len(pending) else "complete", "completed": len(rows), "assigned": len(assigned), "seconds": time.perf_counter() - started})
    print(output)


def logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1 - 1e-6)
    return math.log(value / (1 - value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        ids = [i for i, value in enumerate(scores) if low <= value < high or (index == bins - 1 and value == 1.0)]
        if ids:
            result += len(ids) / len(scores) * abs(mean(scores[i] for i in ids) - mean(labels[i] for i in ids))
    return result


def fit_temperature(scores: list[float], labels: list[int]) -> float:
    candidates = np.linspace(0.25, 4.0, 151)
    losses = []
    for temperature in candidates:
        probs = [sigmoid(logit(value) / temperature) for value in scores]
        losses.append(-mean(label * math.log(max(prob, 1e-9)) + (1 - label) * math.log(max(1 - prob, 1e-9)) for prob, label in zip(probs, labels)))
    return float(candidates[int(np.argmin(losses))])


def fit_platt(scores: list[float], labels: list[int]) -> Callable[[float], float]:
    if len(set(labels)) < 2:
        value = float(labels[0]) if labels else 0.5
        return lambda _: value
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=20260714).fit(np.asarray([[logit(value)] for value in scores]), np.asarray(labels))
    return lambda value: float(model.predict_proba(np.asarray([[logit(value)]]))[0, 1])


def prepare_action_labels(actions: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        grouped[str(action["query_id"])].append(action)
    rows = []
    for query_id, values in grouped.items():
        fallback = next(row for row in values if row["action_family"] == "fallback")
        baseline = outcomes[str(fallback["action_id"])]
        for action in values:
            if action["action_family"] == "fallback":
                continue
            outcome = outcomes[str(action["action_id"])]
            answer_delta = float(outcome["answer_f1"]) - float(baseline["answer_f1"])
            product_delta = float(outcome["answer_title_product"]) - float(baseline["answer_title_product"])
            rows.append({**action, "answer_safe": int(answer_delta >= -1e-12), "answer_drop": int(answer_delta < -1e-12), "positive_action": int(answer_delta >= -1e-12 and product_delta > 1e-12)})
    return rows


def threshold_for_risk(rows: list[dict[str, Any]], score_key: str, target: float = 0.04) -> float:
    candidates = sorted({float(row[score_key]) for row in rows})
    feasible = []
    for threshold in candidates:
        selected = [row for row in rows if float(row[score_key]) >= threshold]
        if selected and mean(float(row["answer_drop"]) for row in selected) <= target:
            feasible.append((len(selected), -threshold, threshold))
    return max(feasible)[2] if feasible else 1.0


def evaluation_actions() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    actions = read_jsonl(EXTERNAL / "generated_actions_1000.jsonl")
    selector = load_module(V4 / "07_train_nested_selector_v4.py", "v5_2wiki_eval_selector")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        if row["action_family"] != "fallback":
            grouped[int(row["outer_fold"])].append(row)
    scored_by_id = {}
    for fold, rows in grouped.items():
        bundle = joblib.load(V4 / "outputs/scaleup/selector_models" / f"fold_{fold}_selector.joblib")
        safe = selector.probabilities(bundle["safety_model"], rows)
        positive = selector.probabilities(bundle["opportunity_model"], rows)
        for row, safe_value, positive_value in zip(rows, safe, positive):
            value = dict(row)
            value["pred_answer_safe_prob"] = safe_value
            value["pred_positive_prob"] = positive_value
            scored_by_id[str(row["action_id"])] = value
    scored = [scored_by_id.get(str(row["action_id"]), row) for row in actions]
    return scored, {str(row["action_id"]): row for row in read_jsonl(EXTERNAL / "reader/all_action_outcomes.jsonl")}


def external_official_by_action(actions: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    completion_eval = load_module(V4_COMPLETION / "04_evaluate_2wiki_frozen_transfer.py", "v5_2wiki_external_eval")
    official = load_module(V4 / "08_run_official_hotpot_evaluation.py", "v5_2wiki_official")
    hotpot = official.load_official(official.DEFAULT_ARROW)
    dev_selections = read_jsonl(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl")
    dev_actions = {str(row["action_id"]): row for row in read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")}
    train = []
    for selection in dev_selections:
        query_id = str(selection["query_id"])
        for action_id in (f"{query_id}::v4::fallback", str(selection["action_id"])):
            train.extend(official.context_instances(query_id, dev_actions[action_id], hotpot[query_id]))
    support_model = official.fit(train)
    source = {str(row["query_id"]): row for row in read_json(EXTERNAL / "2wiki_frozen_1000.json")}
    result = {}
    for action in actions:
        query_id, action_id = str(action["query_id"]), str(action["action_id"])
        item = source[query_id]
        instances = completion_eval.external_instances(query_id, action, item, official)
        pred_support = official.support_set(instances, official.score(support_model, instances), 0.7)
        gold = {(official.normalize_title(str(value[0])), int(value[1])) for value in item.get("supporting_facts", [])}
        result[action_id] = official.official_metrics(outcomes[action_id]["prediction"], item["answer"], pred_support, gold)
    return result


def select_actions(actions: list[dict[str, Any]], transform: Callable[[float], float], threshold: float) -> dict[str, dict[str, Any]]:
    selector = load_module(V4 / "07_train_nested_selector_v4.py", "v5_2wiki_calibrated_selector")
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for action in actions:
        if action["action_family"] != "fallback":
            grouped[int(action["outer_fold"])][str(action["query_id"])].append(action)
    selected = {}
    for fold, queries in grouped.items():
        bundle = joblib.load(V4 / "outputs/scaleup/selector_models" / f"fold_{fold}_selector.joblib")
        cfg = bundle["config"]
        candidates = []
        for query_id, rows in queries.items():
            valid = [row for row in rows if transform(float(row["pred_answer_safe_prob"])) >= threshold and float(row["pred_positive_prob"]) >= float(cfg["positive_threshold"])]
            if valid:
                candidates.append(max(valid, key=lambda row: (float(row["pred_positive_prob"]), transform(float(row["pred_answer_safe_prob"])))))
        budget = min(len(candidates), int(round(float(cfg["coverage"]) * len(queries))))
        chosen = sorted(candidates, key=lambda row: (float(row["pred_positive_prob"]), transform(float(row["pred_answer_safe_prob"]))), reverse=True)[:budget]
        selected.update({str(row["query_id"]): row for row in chosen})
    return selected


def calibrate_stage(args: argparse.Namespace) -> None:
    cfg = config()["two_wiki_calibration"]
    calibration_actions = read_jsonl(OUT / "calibration_actions_union.jsonl")
    calibration_outcomes = {str(row["action_id"]): row for path in sorted((OUT / "reader").glob("outcomes.shard*-of-*.jsonl")) for row in iter_jsonl(path)}
    if len(calibration_outcomes) != len(calibration_actions):
        raise AssertionError(f"Calibration reader incomplete: {len(calibration_outcomes)}/{len(calibration_actions)}")
    calibration_rows = prepare_action_labels(calibration_actions, calibration_outcomes)
    sets = read_json(OUT / "calibration_sets.json")
    eval_actions, eval_outcomes = evaluation_actions()
    eval_labels = prepare_action_labels(eval_actions, eval_outcomes)
    official_by_action = external_official_by_action(eval_actions, eval_outcomes)
    baseline_action = {str(row["query_id"]): row for row in eval_actions if row["action_family"] == "fallback"}
    zero_shot = read_json(EXTERNAL / "external_validation_results.json")
    results = []
    seed_rows = []
    for seed in cfg["seeds"]:
        for k in cfg["k_shots"]:
            query_ids = set(sets[str(seed)][str(k)])
            train = [row for row in calibration_rows if str(row["query_id"]) in query_ids]
            raw_scores = [float(row["pred_answer_safe_prob"]) for row in train]
            labels = [int(row["answer_safe"]) for row in train]
            temperature = fit_temperature(raw_scores, labels)
            platt = fit_platt(raw_scores, labels)
            transforms: dict[str, tuple[Callable[[float], float], float]] = {
                "threshold_only": (lambda value: value, threshold_for_risk(train, "pred_answer_safe_prob")),
                "temperature_scaling": (lambda value, t=temperature: sigmoid(logit(value) / t), 0.5),
                "platt_scaling": (platt, 0.5),
            }
            for row in train:
                row["platt_score"] = platt(float(row["pred_answer_safe_prob"]))
            transforms["risk_constrained"] = (platt, threshold_for_risk(train, "platt_score", cfg["target_selected_answer_drop_rate"]))
            for method, (transform, threshold) in transforms.items():
                selected = select_actions(eval_actions, transform, threshold)
                per_query = []
                for query_id, fallback in baseline_action.items():
                    action = selected.get(query_id, fallback)
                    values = official_by_action[str(action["action_id"])]
                    baseline_values = official_by_action[str(fallback["action_id"])]
                    per_query.append({"query_id": query_id, "selected": query_id in selected, **values, **{f"baseline_{metric}": baseline_values[metric] for metric in METRICS}})
                selected_rows = [row for row in per_query if row["selected"]]
                safety_scores = [transform(float(row["pred_answer_safe_prob"])) for row in eval_labels]
                safety_labels = [int(row["answer_safe"]) for row in eval_labels]
                payload = {
                    "seed": seed,
                    "k": k,
                    "method": method,
                    "threshold": threshold,
                    "temperature": temperature if method == "temperature_scaling" else None,
                    "coverage": len(selected_rows) / len(per_query),
                    "selected_count": len(selected_rows),
                    "answer_drop_rate": mean(float(row["answer_f1"] < row["baseline_answer_f1"] - 1e-12) for row in selected_rows) if selected_rows else 0.0,
                    "ece": ece(safety_scores, safety_labels),
                    "brier": mean((score - label) ** 2 for score, label in zip(safety_scores, safety_labels)),
                    "metrics": {metric: mean(float(row[metric]) for row in per_query) for metric in METRICS},
                    "deltas": {metric: mean(float(row[metric]) - float(row[f"baseline_{metric}"]) for row in per_query) for metric in METRICS},
                }
                results.append(payload)
                seed_rows.append(payload)
    zero = {"method": "zero_shot_frozen", "k": 0, "seed": None, "coverage": zero_shot["selector"]["coverage"], "answer_drop_rate": zero_shot["selector"]["selected_answer_drop_rate"], "metrics": zero_shot["metrics"]["v4_frozen_transfer"], "deltas": zero_shot["deltas"], "ece": "[NOT AVAILABLE]", "brier": "[NOT AVAILABLE]"}
    summaries = {}
    for k in cfg["k_shots"]:
        summaries[str(k)] = {}
        for method in cfg["methods"][1:]:
            rows = [row for row in results if row["k"] == k and row["method"] == method]
            summaries[str(k)][method] = {
                "seeds": len(rows),
                "coverage_mean": mean(row["coverage"] for row in rows),
                "answer_drop_rate_mean": mean(row["answer_drop_rate"] for row in rows),
                "answer_f1_mean": mean(row["metrics"]["answer_f1"] for row in rows),
                "sp_f1_mean": mean(row["metrics"]["sp_f1"] for row in rows),
                "joint_f1_mean": mean(row["metrics"]["joint_f1"] for row in rows),
                "answer_f1_delta_mean": mean(row["deltas"]["answer_f1"] for row in rows),
                "joint_f1_delta_mean": mean(row["deltas"]["joint_f1"] for row in rows),
                "ece_mean": mean(row["ece"] for row in rows),
                "brier_mean": mean(row["brier"] for row in rows),
            }
    payload = {"status": "complete", "zero_shot": zero, "seed_results": results, "summary": summaries, "evaluation_n": 1000, "calibration_split": "2Wiki train", "evaluation_outcomes_used_for_calibration": False}
    write_json(OUT / "calibration_results.json", payload)
    import csv

    with (OUT / "calibration_seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["seed", "k", "method", "threshold", "temperature", "coverage", "selected_count", "answer_drop_rate", "ece", "brier", "answer_f1", "sp_f1", "joint_f1", "answer_f1_delta", "joint_f1_delta"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in seed_rows:
            writer.writerow({**{key: row.get(key) for key in fields}, "answer_f1": row["metrics"]["answer_f1"], "sp_f1": row["metrics"]["sp_f1"], "joint_f1": row["metrics"]["joint_f1"], "answer_f1_delta": row["deltas"]["answer_f1"], "joint_f1_delta": row["deltas"]["joint_f1"]})
    plot_calibration(payload)
    write_report(payload)
    print(json.dumps({"status": "complete", "summary": summaries}, ensure_ascii=False, indent=2))


def plot_calibration(payload: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt

        figures = HERE / "outputs" / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(5.5, 3.6))
        for method in config()["two_wiki_calibration"]["methods"][1:]:
            rows = sorted((row for row in payload["seed_results"] if row["method"] == method), key=lambda row: row["coverage"])
            axis.scatter([row["coverage"] for row in rows], [row["answer_drop_rate"] for row in rows], s=18, alpha=0.65, label=method)
        axis.axhline(0.04, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Selection coverage")
        axis.set_ylabel("Selected answer-drop rate")
        axis.legend(fontsize=7, frameon=False)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "2wiki_risk_coverage.pdf", bbox_inches="tight")
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(5.5, 3.6))
        methods = config()["two_wiki_calibration"]["methods"][1:]
        for method in methods:
            values = [payload["summary"][str(k)][method]["ece_mean"] for k in config()["two_wiki_calibration"]["k_shots"]]
            axis.plot(config()["two_wiki_calibration"]["k_shots"], values, marker="o", label=method)
        axis.set_xlabel("Calibration examples")
        axis.set_ylabel("Safety ECE")
        axis.legend(fontsize=7, frameon=False)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / "2wiki_calibration_curve.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        write_text(HERE / "outputs/figures/2wiki_plot_error.txt", str(exc))


def write_report(payload: dict[str, Any] | None = None) -> None:
    payload = payload or read_json(OUT / "calibration_results.json")
    lines = ["# 2Wiki Few-Shot Safety Calibration Report", "", "## Zero-Shot Frozen Transfer", ""]
    if not payload:
        lines.append("[NEEDS MEASUREMENT]")
    else:
        zero = payload["zero_shot"]
        lines.extend([f"The unchanged HotpotQA gate selects {zero['coverage']:.1%} of the fixed 1,000-query 2Wiki sample and has a selected answer-drop rate of {zero['answer_drop_rate']:.2%}. This result remains the zero-shot transfer result and is not overwritten by calibration.", "", "## Few-Shot Target Calibration", "", "| K | Method | Coverage | Answer-drop | Answer F1 | SP F1 | Joint F1 | ECE | Brier |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
        for k, methods in payload["summary"].items():
            for method, row in methods.items():
                lines.append(f"| {k} | {method} | {row['coverage_mean']:.3f} | {row['answer_drop_rate_mean']:.3f} | {row['answer_f1_mean']:.4f} | {row['sp_f1_mean']:.4f} | {row['joint_f1_mean']:.4f} | {row['ece_mean']:.4f} | {row['brier_mean']:.4f} |")
        successes = [(int(k), method, row) for k, methods in payload["summary"].items() for method, row in methods.items() if row["answer_drop_rate_mean"] <= 0.04 and row["answer_f1_delta_mean"] >= 0 and row["joint_f1_delta_mean"] > 0]
        lines.extend(["", "## Claim Boundary", ""])
        if successes:
            best = min(successes, key=lambda value: (value[0], -value[2]["joint_f1_delta_mean"]))
            lines.append(f"The smallest successful calibration setting is K={best[0]} with `{best[1]}`. This is a few-shot target-calibration result, not zero-shot generalization.")
        else:
            lines.append("No setting simultaneously met the <=4% answer-drop, non-decreasing Answer F1, and positive Joint F1 criteria. Safety transfer remains unresolved.")
    text = "\n".join(lines)
    write_text(HERE / "reports/2wiki_calibration_report.md", text)
    write_text(HERE / "2wiki_calibration_report.md", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "reader", "calibrate", "report"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=os.environ.get("V4_FLAN_T5_LARGE", FLAN))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "reader":
        reader_stage(args)
    elif args.stage == "calibrate":
        calibrate_stage(args)
    else:
        write_report()


if __name__ == "__main__":
    main()
