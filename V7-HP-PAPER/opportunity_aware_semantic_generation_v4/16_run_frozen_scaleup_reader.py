#!/usr/bin/env python3
"""Run one unchanged reader on baseline and frozen-selector scale-up contexts."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from v4_common import OUTPUTS, answer_access, answer_scores, ensure_layout, read_json, read_jsonl, title_metrics, write_json, write_jsonl


FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
UNIFIEDQA = "/home/iiserver31/.cache/huggingface/hub/models--allenai--unifiedqa-v2-t5-large-1363200/snapshots/1d3b8e13b29dbd161494b0b15428378f4713c418"


def flan_prompt(question: str, docs: list[dict], max_chars: int = 3200) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, start=1))
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"
    )


def unifiedqa_prompt(question: str, docs: list[dict], max_chars: int = 3200) -> str:
    context = " ".join(f"{doc['title']}: {doc['text']}" for doc in docs)[:max_chars]
    return f"{question} \n {context}"


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    metrics = ["answer_em", "answer_f1", "answer_access_at_5", "title_recall", "title_f1", "answer_title_product"]
    return {
        method: {metric: sum(float(row[metric]) for row in values) / len(values) for metric in metrics}
        for method, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=["flan", "unifiedqa"], required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    scale_dir = OUTPUTS / "scaleup"
    selector_manifest = read_json(scale_dir / "frozen_selector_manifest.json")
    if selector_manifest.get("status") != "pass" or selector_manifest.get("thresholds_retuned"):
        raise AssertionError("Frozen selector deployment audit failed")

    source = {str(row["_id"]): row for row in read_json(scale_dir / "same_source_hotpot_validation_3000.json")}
    actions = {str(row["action_id"]): row for row in read_jsonl(scale_dir / "generated_actions_3000.jsonl")}
    selections = read_jsonl(scale_dir / "frozen_selector_selections_3000.jsonl")
    pairs = []
    for selection in selections:
        query_id = str(selection["query_id"])
        pairs.extend([
            {"query_id": query_id, "method": "baseline", "action_id": f"{query_id}::v4scale::fallback"},
            {"query_id": query_id, "method": "v4_selected", "action_id": str(selection["action_id"])},
        ])

    output_dir = scale_dir / "readers" / args.reader
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "per_query.jsonl"
    existing = read_jsonl(output_path) if args.resume and output_path.exists() else []
    done = {(str(row["query_id"]), str(row["method"])) for row in existing}
    if output_path.exists() and not args.resume:
        output_path.unlink()
    pending = [row for row in pairs if (row["query_id"], row["method"]) not in done]

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = args.model_path or (FLAN if args.reader == "flan" else UNIFIEDQA)
    runtime = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=args.reader == "flan")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(runtime)
    model.eval()
    prompt_fn = flan_prompt if args.reader == "flan" else unifiedqa_prompt
    write_json(output_dir / "environment_manifest.json", {
        "reader": args.reader,
        "model_path": model_path,
        "device": args.device,
        "batch_size": args.batch_size,
        "prompt_unchanged_from_v4_1000": True,
        "generation": {"max_new_tokens": 32, "num_beams": 1, "do_sample": False},
        "thresholds_retuned": False,
        "started_at_epoch": time.time(),
    })

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompts = [prompt_fn(source[row["query_id"]]["question"], actions[row["action_id"]]["context_docs"]) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(runtime)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        output_rows = []
        for spec, prediction in zip(batch, predictions):
            item = source[spec["query_id"]]
            action = actions[spec["action_id"]]
            exact_match, answer_f1 = answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = title_metrics(action["context_titles"], item["supporting_titles"])
            output_rows.append({
                **spec,
                "prediction": prediction.strip(),
                "answer_em": exact_match,
                "answer_f1": answer_f1,
                "answer_access_at_5": answer_access(item["answer"], action["context_docs"]),
                "title_recall": title_recall,
                "title_f1": title_f1,
                "answer_title_product": answer_f1 * title_f1,
                "context_titles": action["context_titles"],
            })
        write_jsonl(output_path, output_rows, mode="a")
        completed = len(existing) + start + len(batch)
        if completed % 240 < args.batch_size:
            write_json(output_dir / "progress.json", {"completed": completed, "total": len(pairs), "epoch": time.time()})

    rows = read_jsonl(output_path)
    summary = summarize(rows)
    payload = {
        "status": "complete",
        "reader": args.reader,
        "n_queries": len(selections),
        "n_rows": len(rows),
        "metrics": summary,
        "deltas": {metric: summary["v4_selected"][metric] - summary["baseline"][metric] for metric in summary["baseline"]},
        "thresholds_retuned": False,
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
