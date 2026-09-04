#!/usr/bin/env python3
"""Validate frozen v4 selected contexts with a second local QA reader."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from statistics import mean

from v4_common import OUTPUTS, REPORTS, answer_scores, ensure_layout, load_source_examples, normalize_answer, read_json, read_jsonl, write_json, write_jsonl


DEFAULT_READER = "/home/iiserver31/.cache/huggingface/hub/models--allenai--unifiedqa-v2-t5-large-1363200/snapshots/1d3b8e13b29dbd161494b0b15428378f4713c418"


def prompt(question: str, docs: list[dict]) -> str:
    context = " ".join(f"{doc['title']}: {doc['text']}" for doc in docs)[:3200]
    return f"{question} \n {context}"


def answer_pr(prediction: str, gold: str) -> tuple[float, float]:
    predicted, truth = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = Counter(predicted) & Counter(truth)
    overlap = sum(common.values())
    return (
        overlap / len(predicted) if predicted else float(predicted == truth),
        overlap / len(truth) if truth else float(predicted == truth),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("V4_SECOND_READER", DEFAULT_READER))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    nested = read_json(OUTPUTS / "nested_selector/v4_nested_summary.json")
    official = read_json(OUTPUTS / "official_metrics/official_hotpotqa_summary.json")
    if nested.get("status") != "complete" or official.get("status") != "complete":
        reason = "Multi-reader evaluation requires a completed nested selector and official metric stage."
        write_json(OUTPUTS / "multi_reader/multi_reader_summary.json", {"status": "skipped_by_upstream_gate", "reason": reason})
        (REPORTS / "multi_reader_report.md").write_text(f"# Multi-Reader Validation\n\nStatus: skipped. {reason}\n", encoding="utf-8")
        print(json.dumps({"status": "skipped_by_upstream_gate"}, indent=2))
        return

    selections = read_jsonl(OUTPUTS / "nested_selector/v4_nested_per_query.jsonl")
    actions = {str(row["action_id"]): row for row in read_jsonl(OUTPUTS / "generated_actions/v4_outer_test_actions.jsonl")}
    source = load_source_examples()
    pairs = []
    for selection in selections:
        query_id = str(selection["query_id"])
        pairs.extend([
            {"query_id": query_id, "method": "baseline", "action_id": f"{query_id}::v4::fallback"},
            {"query_id": query_id, "method": "v4_selected", "action_id": str(selection["action_id"])},
        ])
    prediction_path = OUTPUTS / "multi_reader/unifiedqa_per_query.jsonl"
    if args.reuse_predictions and prediction_path.exists():
        rows = read_jsonl(prediction_path)
    else:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = torch.device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.float16).to(device)
        model.eval()
        rows = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            prompts = [prompt(source[row["query_id"]]["question"], actions[row["action_id"]]["context_docs"]) for row in batch]
            encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for spec, prediction in zip(batch, predictions):
                exact_match, f1 = answer_scores(prediction.strip(), source[spec["query_id"]]["answer"])
                precision, recall = answer_pr(prediction.strip(), source[spec["query_id"]]["answer"])
                rows.append({**spec, "prediction": prediction.strip(), "answer_em": exact_match, "answer_f1": f1, "answer_precision": precision, "answer_recall": recall})
        write_jsonl(prediction_path, rows)
    if rows and "answer_precision" not in rows[0]:
        for row in rows:
            precision, recall = answer_pr(row["prediction"], source[row["query_id"]]["answer"])
            row["answer_precision"], row["answer_recall"] = precision, recall
        write_jsonl(prediction_path, rows)

    official_rows = {(str(row["query_id"]), str(row["method"])): row for row in read_jsonl(OUTPUTS / "official_metrics/official_hotpotqa_per_query.jsonl")}
    for row in rows:
        support = official_rows[(str(row["query_id"]), str(row["method"]))]
        joint_precision = float(row["answer_precision"]) * float(support["sp_precision"])
        joint_recall = float(row["answer_recall"]) * float(support["sp_recall"])
        row["sp_f1"] = float(support["sp_f1"])
        row["joint_f1"] = 2 * joint_precision * joint_recall / (joint_precision + joint_recall) if joint_precision + joint_recall else 0.0
    write_jsonl(prediction_path, rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    unified = {method: {metric: mean(row[metric] for row in values) for metric in ("answer_em", "answer_f1", "sp_f1", "joint_f1")} for method, values in grouped.items()}
    delta = unified["v4_selected"]["answer_f1"] - unified["baseline"]["answer_f1"]
    joint_delta = unified["v4_selected"]["joint_f1"] - unified["baseline"]["joint_f1"]
    by_query = defaultdict(dict)
    for row in rows:
        by_query[str(row["query_id"])][str(row["method"])] = row
    answer_drop_rate = mean(values["v4_selected"]["answer_f1"] < values["baseline"]["answer_f1"] - 1e-12 for values in by_query.values())
    flan_delta = nested["deltas"]["answer_f1"]
    payload = {
        "status": "complete",
        "reader_1": "google/flan-t5-large",
        "reader_2": "allenai/unifiedqa-v2-t5-large-1363200",
        "flan_answer_f1_delta": flan_delta,
        "unifiedqa": unified,
        "unifiedqa_answer_f1_delta": delta,
        "unifiedqa_joint_f1_delta": joint_delta,
        "unifiedqa_answer_drop_rate": answer_drop_rate,
        "support_f1_delta": unified["v4_selected"]["sp_f1"] - unified["baseline"]["sp_f1"],
        "direction_consistent": (flan_delta >= 0) == (delta >= 0),
        "systematic_answer_degradation": flan_delta < 0 and delta < 0,
    }
    write_json(OUTPUTS / "multi_reader/multi_reader_summary.json", payload)
    (REPORTS / "multi_reader_report.md").write_text(
        f"# Multi-Reader Validation\n\nFLAN answer F1 delta: **{flan_delta:+.4f}**. UnifiedQA answer F1 delta: **{delta:+.4f}**; joint F1 delta: **{joint_delta:+.4f}**; answer-drop rate: **{answer_drop_rate:.2%}**. Direction consistent: **{payload['direction_consistent']}**.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
