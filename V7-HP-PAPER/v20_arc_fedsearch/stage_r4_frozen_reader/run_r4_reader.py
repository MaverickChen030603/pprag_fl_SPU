#!/usr/bin/env python3
"""Run one frozen legacy reader over immutable R4 contexts.

The only reader-visible fields are question and title/text documents from the
materialized context JSONL. Gold answer/support values are loaded only after
generation for offline scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np


V20 = Path(__file__).resolve().parents[1]
V16 = V20.parent / "v16_action_composition"
sys.path.insert(0, str(V16 / "evaluation"))
from eval_common import normalize_title, official_metrics, source_documents, unit_features  # noqa: E402


SPLITS = {
    "2wikimultihopqa": V20 / "stage_r3_probe_route/protocol/2wikimultihopqa/probe_holdout.jsonl",
    "musique": V20 / "stage_r3_probe_route/protocol/musique/probe_holdout.jsonl",
    "hotpotqa": V20 / "stage_r3_probe_route/hotpot_transfer/protocol/probe_holdout.jsonl",
}
MODELS = {
    "flan": "google/flan-t5-large",
    "unifiedqa": "allenai/unifiedqa-v2-t5-large-1363200",
}


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def prompt_flan(question: str, docs: list[dict[str, Any]], max_chars: int) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, 1))
    return f"Answer the question using only the context. Return a short answer.\n\nQuestion: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"


def prompt_unifiedqa(question: str, docs: list[dict[str, Any]], max_chars: int) -> str:
    context = " ".join(f"{doc['title']}: {doc['text']}" for doc in docs)
    return f"{question} \n {context[:max_chars]}"


def payload_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def support_prediction(checkpoint: dict[str, Any], question: str, docs: list[dict[str, Any]], source_row: dict[str, Any], dataset: str) -> set[tuple[str, int]]:
    local = {doc["doc_id"]: doc for doc in source_documents(source_row, dataset)}
    instances = []
    for doc_rank, doc in enumerate(docs):
        exact = local.get(doc["doc_id"])
        if dataset == "musique":
            identity = ("paragraph", int(exact["paragraph_idx"])) if exact else ("document", int(hashlib.sha1(doc["doc_id"].encode()).hexdigest()[:8], 16))
            instances.append({"identity": identity, "features": unit_features(question, doc["title"], doc["text"], doc_rank, 0, 1)})
            continue
        sentences = exact.get("sentences", []) if exact else [value.strip() for value in re.split(r"(?<=[.!?])\s+", str(doc["text"])) if value.strip()]
        sentences = sentences or [str(doc["text"])]
        for sent_id, sentence in enumerate(sentences):
            instances.append({"identity": (normalize_title(doc["title"]), sent_id), "features": unit_features(question, doc["title"], sentence, doc_rank, sent_id, len(sentences))})
    if not instances:
        return set()
    probabilities = checkpoint["model"].predict_proba(np.asarray([row["features"] for row in instances]))[:, 1]
    ranked = sorted(zip(instances, probabilities), key=lambda pair: pair[1], reverse=True)
    selected = [row for row, score in ranked if score >= checkpoint["threshold"]][:6]
    if len(selected) < int(checkpoint.get("minimum_predictions", 2)):
        selected = [row for row, _ in ranked[:int(checkpoint.get("minimum_predictions", 2))]]
    return {tuple(row["identity"]) for row in selected}


def support_docs(source_row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {doc["doc_id"] for doc in source_documents(source_row, dataset) if any(int(p.get("idx", -1)) == int(doc["paragraph_idx"]) and p.get("is_supporting", False) for p in source_row.get("paragraphs", []))}
    titles = source_row.get("supporting_facts", {}).get("title", []) if isinstance(source_row.get("supporting_facts"), dict) else [value[0] for value in source_row.get("supporting_facts", [])]
    return {doc["doc_id"] for doc in source_documents(source_row, dataset) if normalize_title(doc["title"]) in {normalize_title(x) for x in titles}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=tuple(MODELS), required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--smoke-per-dataset", type=int, default=0,
                        help="deterministically select this many query IDs per dataset; engineering smoke only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    inputs = list(rows(args.contexts))
    if args.smoke_per_dataset:
        selected: dict[str, set[str]] = {}
        for dataset in sorted({str(row["dataset"]) for row in inputs}):
            query_ids = sorted({str(row["query_id"]) for row in inputs if row["dataset"] == dataset},
                               key=lambda value: hashlib.sha256(f"r4-p1-smoke|{dataset}|{value}".encode()).hexdigest())
            selected[dataset] = set(query_ids[:args.smoke_per_dataset])
        inputs = [row for row in inputs if str(row["query_id"]) in selected[str(row["dataset"])]]
    source = {dataset: {qid(row): row for row in rows(path)} for dataset, path in SPLITS.items()}
    missing = [(row["dataset"], row["query_id"]) for row in inputs if row["query_id"] not in source[row["dataset"]]]
    if missing:
        raise ValueError(f"context query absent from frozen split: {missing[:3]}")
    existing = list(rows(args.output)) if args.resume and args.output.exists() else []
    done = {(str(row["dataset"]), str(row["query_id"]), str(row["method"])) for row in existing}
    pending = [row for row in inputs if (row["dataset"], row["query_id"], row["method"]) not in done]
    if args.output.exists() and not args.resume:
        args.output.unlink()

    model_name = MODELS[args.reader]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=args.reader == "flan")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True, torch_dtype=dtype).to(torch.device(args.device))
    model.eval()
    checkpoints = {dataset: joblib.load(V16 / "evaluation/checkpoints" / f"{dataset}_support.joblib") for dataset in SPLITS}
    prompt_fn = prompt_flan if args.reader == "flan" else prompt_unifiedqa
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompts = [prompt_fn(str(row["question"]), row["reader_context_docs"], args.max_chars) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(args.device)
        call_start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        elapsed = (time.perf_counter() - call_start) / len(batch)
        with args.output.open("a", encoding="utf-8") as handle:
            for row, prompt, prediction in zip(batch, prompts, predictions):
                dataset, query_id = row["dataset"], row["query_id"]
                gold_row = source[dataset][query_id]
                support = support_prediction(checkpoints[dataset], str(row["question"]), row["reader_context_docs"], gold_row, dataset)
                metrics = official_metrics(prediction.strip(), gold_row, support, dataset)
                gold_ids = support_docs(gold_row, dataset)
                payload = {
                    "query_id": query_id, "question": row["question"], "dataset": dataset, "method": row["method"], "reader": args.reader,
                    "retrieved_doc_ids": row["retrieved_doc_ids"], "reader_context_doc_ids": row["reader_context_doc_ids"], "context_hash": row["context_hash"],
                    "predicted_answer": prediction.strip(), "gold_answer": str(gold_row.get("answer", "")),
                    "predicted_support": [list(value) for value in sorted(support)], "gold_support": [list(value) for value in sorted(__import__("eval_common").gold_support(gold_row, dataset))],
                    "retrieval_complete_support": int(bool(gold_ids) and gold_ids.issubset(set(row["reader_context_doc_ids"]))),
                    "reader_output_hash": hashlib.sha256(prediction.strip().encode()).hexdigest(), "reader_input_hash": payload_hash(prompt),
                    "runtime_seconds": elapsed, "error_code": "", **metrics,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        completed = len(existing) + start + len(batch)
        if completed == len(existing) + len(batch) or completed % 100 < len(batch):
            print(json.dumps({"reader": args.reader, "completed": completed, "total": len(inputs), "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    print(json.dumps({"status": "complete", "reader": args.reader, "rows": len(inputs), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
