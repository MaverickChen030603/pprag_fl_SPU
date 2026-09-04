#!/usr/bin/env python3
"""Run the frozen FLAN-T5-large reader over baseline and v3 action contexts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v3_common import OUTPUTS, answer_access, answer_scores, ensure_layout, load_source_examples, read_jsonl, title_metrics, write_json, write_jsonl


DEFAULT_MODEL = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
MODEL_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"


def prompt(question: str, docs: list[dict[str, Any]], max_chars: int = 3200) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, start=1))
    context = context[:max_chars]
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )


def environment_manifest(model_path: str) -> dict[str, Any]:
    import torch
    import transformers

    return {
        "model_id": "google/flan-t5-large",
        "model_path": model_path,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "prompt": "Answer the question using only the context. Return a short answer.",
        "context_char_limit": 3200,
        "tokenizer_input_limit": 1024,
        "max_new_tokens": 32,
        "num_beams": 1,
        "do_sample": False,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    positive_count = 0
    answer_safe_count = 0
    covered = 0
    family_positive_actions: Counter[str] = Counter()
    family_positive_queries: dict[str, set[str]] = defaultdict(set)
    per_query: list[dict[str, Any]] = []
    for query_id, values in by_query.items():
        baseline = next(row for row in values if row["action_family"] == "fallback")
        actions = [row for row in values if row["action_family"] != "fallback"]
        positives = []
        for row in actions:
            row["answer_f1_delta"] = row["answer_f1"] - baseline["answer_f1"]
            row["title_recall_delta"] = row["title_recall"] - baseline["title_recall"]
            row["title_f1_delta"] = row["title_f1"] - baseline["title_f1"]
            row["answer_title_product_delta"] = row["answer_title_product"] - baseline["answer_title_product"]
            row["answer_safe"] = row["answer_f1_delta"] >= -1e-12
            row["positive_action"] = bool(row["answer_safe"] and row["answer_title_product_delta"] > 1e-12 and (row["title_recall_delta"] > 1e-12 or row["title_f1_delta"] >= -1e-12))
            answer_safe_count += int(row["answer_safe"])
            if row["positive_action"]:
                positive_count += 1
                positives.append(row)
                family_positive_actions[row["action_family"]] += 1
                family_positive_queries[row["action_family"]].add(query_id)
        covered += int(bool(positives))
        per_query.append({
            "query_id": query_id,
            "num_actions": len(actions),
            "num_positive_actions": len(positives),
            "positive_families": sorted({row["action_family"] for row in positives}),
            "best_product_delta": max((row["answer_title_product_delta"] for row in actions), default=0.0),
        })
    n_queries = len(by_query)
    n_actions = sum(1 for row in rows if row["action_family"] != "fallback")
    coverage = covered / n_queries if n_queries else 0.0
    gate = "strong" if coverage >= 0.40 else "meaningful" if coverage >= 0.30 else "unlikely" if coverage < 0.25 else "borderline"
    return {
        "status": "complete",
        "num_queries": n_queries,
        "num_effective_actions": n_actions,
        "positive_actions": positive_count,
        "positive_action_rate": positive_count / n_actions if n_actions else 0.0,
        "answer_safe_actions": answer_safe_count,
        "answer_safe_action_rate": answer_safe_count / n_actions if n_actions else 0.0,
        "queries_with_positive_action": covered,
        "queries_without_positive_action": n_queries - covered,
        "positive_query_coverage": coverage,
        "v2_positive_query_coverage": 0.203,
        "coverage_delta_vs_v2": coverage - 0.203,
        "pre_registered_gate": gate,
        "proceed_to_nested_selector": coverage >= 0.25,
        "family_positive_action_counts": dict(family_positive_actions),
        "family_positive_query_counts": {family: len(query_ids) for family, query_ids in family_positive_queries.items()},
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("V3_FLAN_T5_LARGE", DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    action_path = OUTPUTS / "candidate_generation/v3_candidate_actions.jsonl"
    actions = read_jsonl(action_path)
    source = load_source_examples()
    query_ids = sorted({row["query_id"] for row in actions})
    if args.max_queries:
        query_ids = query_ids[:args.max_queries]
    query_set = set(query_ids)
    actions = [row for row in actions if row["query_id"] in query_set]
    output_path = OUTPUTS / "action_outcomes/v3_action_reader_outputs.jsonl"
    done: set[tuple[str, str]] = set()
    existing: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        existing = read_jsonl(output_path)
        done = {(str(row["query_id"]), str(row["action_id"])) for row in existing}
    elif output_path.exists():
        output_path.unlink()

    manifest = environment_manifest(args.model_path)
    manifest.update({"batch_size": args.batch_size, "device_argument": args.device, "started_at_epoch": time.time()})
    write_json(OUTPUTS / "action_outcomes/reader_environment_manifest.json", manifest)
    runtime_device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dtype = torch.float16 if runtime_device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=dtype).to(runtime_device)
    model.eval()

    pending = [row for row in actions if (row["query_id"], row["action_id"]) not in done]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompts = [prompt(row["question"], row["context_docs"]) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(runtime_device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        output_rows: list[dict[str, Any]] = []
        for action, prediction in zip(batch, predictions):
            item = source[action["query_id"]]
            answer_em, answer_f1 = answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = title_metrics(action["context_titles"], item["supporting_titles"])
            output_rows.append({
                "query_id": action["query_id"],
                "action_id": action["action_id"],
                "action_family": action["action_family"],
                "action_name": action["action_name"],
                "prediction": prediction.strip(),
                "answer_em": answer_em,
                "answer_f1": answer_f1,
                "answer_access_at_5": answer_access(item["answer"], action["context_docs"]),
                "title_recall": title_recall,
                "title_f1": title_f1,
                "answer_title_product": answer_f1 * title_f1,
                "context_doc_ids": action["context_doc_ids"],
                "context_titles": action["context_titles"],
                "inference_safe_features": action["inference_safe_features"],
            })
        write_jsonl(output_path, output_rows, mode="a")
        completed = len(existing) + start + len(batch)
        if completed % max(args.batch_size, 120) < args.batch_size:
            write_json(OUTPUTS / "action_outcomes/v3_reader_progress.json", {"completed": completed, "total": len(actions), "epoch": time.time()})

    all_rows = read_jsonl(output_path)
    summary = summarize(all_rows)
    write_jsonl(output_path, all_rows)
    write_json(OUTPUTS / "action_outcomes/v3_action_outcome_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ["num_queries", "positive_query_coverage", "coverage_delta_vs_v2", "pre_registered_gate", "proceed_to_nested_selector"]}, indent=2))


if __name__ == "__main__":
    main()

