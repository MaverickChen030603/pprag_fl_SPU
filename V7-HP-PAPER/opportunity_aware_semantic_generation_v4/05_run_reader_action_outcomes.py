#!/usr/bin/env python3
"""Evaluate frozen v4 outer-test actions with the v2/v3 FLAN-T5-large reader."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from typing import Any

from v4_common import (
    OUTPUTS, answer_access, answer_scores, ensure_layout, load_source_examples, read_json,
    read_jsonl, title_metrics, v2_main_positive_query_ids, v3_outcomes_path, write_json, write_jsonl,
)


DEFAULT_MODEL = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
MODEL_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"


def prompt(question: str, docs: list[dict[str, Any]], max_chars: int = 3200) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, start=1))
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"
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


def frozen_v3_positive_query_ids() -> set[str]:
    return {
        str(row["query_id"]) for row in read_jsonl(v3_outcomes_path())
        if row.get("action_family") != "fallback" and bool(row.get("positive_action"))
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    v2_positive = v2_main_positive_query_ids()
    v3_positive = frozen_v3_positive_query_ids()
    ceiling = set(read_json(OUTPUTS / "audits/v3_ceiling_aware_opportunity.json")["ceiling_query_ids"])
    all_query_ids = set(grouped)
    non_ceiling = all_query_ids - ceiling
    positive_actions = 0
    answer_safe_actions = 0
    covered: set[str] = set()
    family_positive_queries: dict[str, set[str]] = defaultdict(set)
    family_positive_actions: Counter[str] = Counter()
    per_query = []
    new_action_count = 0
    for query_id, values in grouped.items():
        baseline = next(row for row in values if row["action_family"] == "fallback")
        actions = [row for row in values if row["action_family"] != "fallback"]
        positives = []
        for row in actions:
            row["answer_f1_delta"] = float(row["answer_f1"]) - float(baseline["answer_f1"])
            row["title_recall_delta"] = float(row["title_recall"]) - float(baseline["title_recall"])
            row["title_f1_delta"] = float(row["title_f1"]) - float(baseline["title_f1"])
            row["answer_title_product_delta"] = float(row["answer_title_product"]) - float(baseline["answer_title_product"])
            row["answer_safe"] = row["answer_f1_delta"] >= -1e-12
            row["positive_action"] = bool(
                row["answer_safe"]
                and row["answer_title_product_delta"] > 1e-12
                and (row["title_recall_delta"] > 1e-12 or row["title_f1_delta"] >= -1e-12)
            )
            answer_safe_actions += int(row["answer_safe"])
            new_action_count += int(bool(row.get("is_new_vs_v3_action_table")))
            if row["positive_action"]:
                positive_actions += 1
                positives.append(row)
                family_positive_queries[str(row["action_family"])].add(query_id)
                family_positive_actions[str(row["action_family"])] += 1
        if positives:
            covered.add(query_id)
        per_query.append({
            "query_id": query_id,
            "num_actions": len(actions),
            "num_positive_actions": len(positives),
            "positive_families": sorted({str(row["action_family"]) for row in positives}),
            "newly_covered_vs_v2": bool(positives and query_id not in v2_positive),
            "newly_covered_vs_v3": bool(positives and query_id not in v3_positive),
        })
    effective_actions = sum(1 for row in rows if row["action_family"] != "fallback")
    covered_non_ceiling = covered & non_ceiling
    new_vs_v2 = covered - v2_positive
    new_vs_v3 = covered - v3_positive
    family_diversity = sum(len(row["positive_families"]) for row in per_query if row["num_positive_actions"] > 0) / max(1, len(covered))
    family_overlap = {
        left: {right: len(family_positive_queries[left] & family_positive_queries[right]) for right in sorted(family_positive_queries)}
        for left in sorted(family_positive_queries)
    }
    return {
        "status": "complete",
        "num_queries": len(grouped),
        "num_effective_actions": effective_actions,
        "num_new_actions_vs_v3_table": new_action_count,
        "positive_actions": positive_actions,
        "positive_action_density": positive_actions / effective_actions if effective_actions else 0.0,
        "answer_safe_actions": answer_safe_actions,
        "answer_safe_action_rate": answer_safe_actions / effective_actions if effective_actions else 0.0,
        "queries_with_positive_action": len(covered),
        "overall_positive_query_coverage": len(covered) / len(grouped),
        "non_ceiling_queries": len(non_ceiling),
        "non_ceiling_positive_queries": len(covered_non_ceiling),
        "non_ceiling_positive_query_coverage": len(covered_non_ceiling) / len(non_ceiling),
        "new_queries_covered_beyond_v2": len(new_vs_v2),
        "new_queries_covered_beyond_v3": len(new_vs_v3),
        "net_coverage_gain_queries_vs_v3": len(covered) - len(v3_positive),
        "marginal_new_query_coverage_vs_v3": len(new_vs_v3) / len(grouped),
        "new_query_efficiency": len(new_vs_v3) / new_action_count if new_action_count else 0.0,
        "positive_actions_per_covered_query": positive_actions / len(covered) if covered else 0.0,
        "family_diversity_per_covered_query": family_diversity,
        "family_positive_action_counts": dict(family_positive_actions),
        "family_positive_query_counts": {family: len(query_ids) for family, query_ids in family_positive_queries.items()},
        "family_positive_query_overlap": family_overlap,
        "covered_query_ids": sorted(covered),
        "new_query_ids_vs_v2": sorted(new_vs_v2),
        "new_query_ids_vs_v3": sorted(new_vs_v3),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("V4_FLAN_T5_LARGE", DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    actions = read_jsonl(OUTPUTS / "generated_actions/v4_outer_test_actions.jsonl")
    if args.max_queries:
        query_ids = sorted({str(row["query_id"]) for row in actions})[:args.max_queries]
        query_set = set(query_ids)
        actions = [row for row in actions if str(row["query_id"]) in query_set]
    source = load_source_examples()
    output_path = OUTPUTS / "action_outcomes/v4_action_outputs.jsonl"
    existing = read_jsonl(output_path) if args.resume and output_path.exists() else []
    done = {(str(row["query_id"]), str(row["action_id"])) for row in existing}
    if output_path.exists() and not args.resume:
        output_path.unlink()

    manifest = environment_manifest(args.model_path)
    manifest.update({"batch_size": args.batch_size, "device_argument": args.device, "started_at_epoch": time.time()})
    write_json(OUTPUTS / "action_outcomes/reader_environment_manifest.json", manifest)
    runtime_device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dtype = torch.float16 if runtime_device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=dtype).to(runtime_device)
    model.eval()

    pending = [row for row in actions if (str(row["query_id"]), str(row["action_id"])) not in done]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompts = [prompt(str(row["question"]), row["context_docs"]) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(runtime_device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        output_rows = []
        for action, prediction in zip(batch, predictions):
            item = source[str(action["query_id"])]
            answer_em, answer_f1 = answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = title_metrics(action["context_titles"], item["supporting_titles"])
            output_rows.append({
                "query_id": action["query_id"],
                "action_id": action["action_id"],
                "outer_fold": action["outer_fold"],
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
                "is_new_vs_v3_action_table": action["is_new_vs_v3_action_table"],
                "generator_score": action["generator_score"],
                "inference_safe_features": action["inference_safe_features"],
            })
        write_jsonl(output_path, output_rows, mode="a")
        completed = len(existing) + start + len(batch)
        if completed % max(args.batch_size, 120) < args.batch_size:
            write_json(OUTPUTS / "action_outcomes/v4_reader_progress.json", {"completed": completed, "total": len(actions), "epoch": time.time()})

    all_rows = read_jsonl(output_path)
    summary = summarize(all_rows)
    write_jsonl(output_path, all_rows)
    write_json(OUTPUTS / "action_outcomes/v4_action_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ["num_queries", "num_effective_actions", "positive_action_density", "overall_positive_query_coverage", "non_ceiling_positive_query_coverage", "new_queries_covered_beyond_v3", "new_query_efficiency"]}, indent=2))


if __name__ == "__main__":
    main()
