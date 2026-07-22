#!/usr/bin/env python3
"""Offline two-reader labeling for frozen V16 trajectory contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from eval_common import normalize_title, official_metrics, source_documents, unit_features


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_flan(question: str, docs: list[dict[str, Any]], max_chars: int) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, 1))
    return f"Answer the question using only the context. Return a short answer.\n\nQuestion: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"


def prompt_unifiedqa(question: str, docs: list[dict[str, Any]], max_chars: int) -> str:
    context = " ".join(f"{doc['title']}: {doc['text']}" for doc in docs)
    return f"{question} \n {context[:max_chars]}"


def choose_contexts(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    # Keep the lowest-depth representation for duplicate ordered contexts so a
    # composed candidate cannot masquerade as synergy when one edit reaches it.
    by_context: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row["context_doc_ids"])
        current = by_context.get(key)
        if current is None or (int(row["depth"]), -float(row.get("cheap_score", 0.0))) < (int(current["depth"]), -float(current.get("cheap_score", 0.0))):
            by_context[key] = row
    values = list(by_context.values())
    required = sorted([row for row in values if int(row["depth"]) <= 1], key=lambda row: (int(row["depth"]), -float(row.get("cheap_score", 0.0)), row["trajectory_id"]))
    if not limit or len(values) <= limit:
        return required + sorted([row for row in values if int(row["depth"]) > 1], key=lambda row: (-float(row.get("cheap_score", 0.0)), int(row["depth"]), row["trajectory_id"]))
    if len(required) > limit:
        raise ValueError(f"context cap {limit} is smaller than baseline plus all single edits ({len(required)})")
    remaining = limit - len(required)
    composed = sorted([row for row in values if 2 <= int(row["depth"]) <= 3], key=lambda row: (-float(row.get("cheap_score", 0.0)), int(row["depth"]), row["trajectory_id"]))
    subsets = sorted([row for row in values if int(row["depth"]) > 3], key=lambda row: (-float(row.get("cheap_score", 0.0)), row["trajectory_id"]))
    subset_budget = min(len(subsets), remaining // 4)
    return required + composed[:remaining - subset_budget] + subsets[:subset_budget]


def support_prediction(checkpoint: dict[str, Any], question: str, docs: list[dict[str, Any]], source_row: dict[str, Any], dataset: str) -> set[tuple[str, int]]:
    local = {doc["doc_id"]: doc for doc in source_documents(source_row, dataset)}
    instances = []
    for doc_rank, doc in enumerate(docs):
        exact = local.get(doc["doc_id"])
        if dataset == "musique":
            negative_id = int(hashlib.sha1(doc["doc_id"].encode("utf-8")).hexdigest()[:8], 16)
            identity = ("paragraph", int(exact["paragraph_idx"])) if exact else ("document", negative_id)
            instances.append({"identity": identity, "features": unit_features(question, doc["title"], doc["text"], doc_rank, 0, 1)})
            continue
        sentences = exact.get("sentences", []) if exact else [value.strip() for value in re.split(r"(?<=[.!?])\s+", str(doc["text"])) if value.strip()]
        sentences = sentences or [str(doc["text"])]
        for sent_id, sentence in enumerate(sentences):
            instances.append({"identity": (normalize_title(doc["title"]), sent_id), "features": unit_features(question, doc["title"], sentence, doc_rank, sent_id, len(sentences))})
    probabilities = checkpoint["model"].predict_proba(np.asarray([row["features"] for row in instances]))[:, 1]
    ranked = sorted(zip(instances, probabilities), key=lambda pair: pair[1], reverse=True)
    selected = [row for row, score in ranked if score >= checkpoint["threshold"]][:6]
    minimum = int(checkpoint.get("minimum_predictions", 2))
    if len(selected) < minimum:
        selected = [row for row, _ in ranked[:minimum]]
    return {tuple(row["identity"]) for row in selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=("flan", "unifiedqa"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--support-predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-contexts-per-query", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    source = {str(row["query_id"]): row for row in read(args.split)}
    pools = {str(row["query_id"]): row for row in read(args.pool)}
    grouped = defaultdict(list)
    for row in read(args.contexts):
        grouped[str(row["query_id"])].append(row)
    specs = [row for query_id in sorted(grouped) for row in choose_contexts(grouped[query_id], args.max_contexts_per_query)]
    existing = read(args.output) if args.resume and args.output.exists() else []
    done = {(str(row["query_id"]), str(row["trajectory_id"])) for row in existing}
    pending = [row for row in specs if (str(row["query_id"]), str(row["trajectory_id"])) not in done]
    if args.output.exists() and not args.resume:
        args.output.unlink()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=args.reader == "flan")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float16).to(torch.device(args.device))
    model.eval()
    checkpoint = joblib.load(args.support_predictor)
    prompt_fn = prompt_flan if args.reader == "flan" else prompt_unifiedqa
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        contexts = []
        for spec in batch:
            docs = {doc["doc_id"]: doc for doc in pools[str(spec["query_id"])]["pool"]}
            contexts.append([docs[doc_id] for doc_id in spec["context_doc_ids"]])
        prompts = [prompt_fn(str(source[str(spec["query_id"])]["question"]), docs, args.max_chars) for spec, docs in zip(batch, contexts)]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        with args.output.open("a", encoding="utf-8") as handle:
            for spec, docs, prediction in zip(batch, contexts, predictions):
                query_id = str(spec["query_id"])
                support = support_prediction(checkpoint, str(source[query_id]["question"]), docs, source[query_id], args.dataset)
                metrics = official_metrics(prediction.strip(), source[query_id], support, args.dataset)
                payload = {
                    "query_id": query_id, "dataset": args.dataset, "reader": args.reader,
                    "trajectory_id": spec["trajectory_id"], "depth": int(spec["depth"]),
                    "candidate_type": spec.get("candidate_type", "trajectory"), "is_baseline": bool(spec.get("is_baseline", False)),
                    "context_doc_ids": spec["context_doc_ids"], "prediction": prediction.strip(),
                    "predicted_support": [list(value) for value in sorted(support)],
                    "hop_count": spec.get("hop_count", "unknown"), "question_type": spec.get("question_type", "unknown"), **metrics,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if start == 0 or (start + len(batch)) % 240 < args.batch_size:
            print(json.dumps({"reader": args.reader, "dataset": args.dataset, "completed": len(existing) + start + len(batch), "total": len(specs)}), flush=True)
    print(json.dumps({"status": "complete", "reader": args.reader, "dataset": args.dataset, "queries": len(grouped), "contexts": len(specs), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
