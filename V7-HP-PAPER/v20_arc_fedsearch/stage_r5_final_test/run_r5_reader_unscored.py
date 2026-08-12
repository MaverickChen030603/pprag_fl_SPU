#!/usr/bin/env python3
"""Run a frozen reader without loading final-test labels or computing metrics."""
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


MODELS = {"flan": "google/flan-t5-large", "unifiedqa": "allenai/unifiedqa-v2-t5-large-1363200"}


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row):
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_flan(question, docs):
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, 1))
    return f"Answer the question using only the context. Return a short answer.\n\nQuestion: {question}\n\nContext:\n{context[:4000]}\n\nAnswer:"


def prompt_unifiedqa(question, docs):
    context = " ".join(f"{doc['title']}: {doc['text']}" for doc in docs)
    return f"{question} \n {context[:4000]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=tuple(MODELS), required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--v16-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.v16_eval))
    from eval_common import normalize_title, source_documents, unit_features
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    contexts = list(rows(args.contexts))
    if len(contexts) != 3600:
        raise ValueError(f"expected 3600 contexts, found {len(contexts)}")
    sources = {dataset: {qid(row): row for row in rows(args.sample_root / f"{dataset}_final_test_inputs_n300.jsonl")} for dataset in ("hotpotqa", "2wikimultihopqa", "musique")}
    existing = list(rows(args.output)) if args.resume and args.output.exists() else []
    done = {(row["dataset"], row["method"], row["query_id"]) for row in existing}
    pending = [row for row in contexts if (row["dataset"], row["method"], row["query_id"]) not in done]
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.reader], local_files_only=True, use_fast=args.reader == "flan")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODELS[args.reader], local_files_only=True, torch_dtype=torch.float16).to(args.device)
    model.eval()
    checkpoints = {dataset: joblib.load(args.v16_eval / f"checkpoints/{dataset}_support.joblib") for dataset in sources}
    prompt_fn = prompt_flan if args.reader == "flan" else prompt_unifiedqa

    def predict_support(dataset, question, docs, source_row):
        local = {doc["doc_id"]: doc for doc in source_documents(source_row, dataset)}
        instances = []
        for doc_rank, doc in enumerate(docs):
            exact = local.get(doc["doc_id"])
            if dataset == "musique":
                identity = ("paragraph", int(exact["paragraph_idx"])) if exact else ("document", int(hashlib.sha1(doc["doc_id"].encode()).hexdigest()[:8], 16))
                instances.append({"identity": identity, "features": unit_features(question, doc["title"], doc["text"], doc_rank, 0, 1)})
                continue
            sentences = exact.get("sentences", []) if exact else [x.strip() for x in re.split(r"(?<=[.!?])\s+", str(doc["text"])) if x.strip()]
            sentences = sentences or [str(doc["text"])]
            for sent_id, sentence in enumerate(sentences):
                instances.append({"identity": (normalize_title(doc["title"]), sent_id), "features": unit_features(question, doc["title"], sentence, doc_rank, sent_id, len(sentences))})
        probability = checkpoints[dataset]["model"].predict_proba(np.asarray([row["features"] for row in instances]))[:, 1]
        ranked = sorted(zip(instances, probability), key=lambda pair: pair[1], reverse=True)
        selected = [row for row, score in ranked if score >= checkpoints[dataset]["threshold"]][:6]
        minimum = int(checkpoints[dataset].get("minimum_predictions", 2))
        if len(selected) < minimum:
            selected = [row for row, _ in ranked[:minimum]]
        return [list(row["identity"]) for row in selected]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompts = [prompt_fn(str(row["question"]), row["reader_context_docs"]) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
        with args.output.open("a", encoding="utf-8") as handle:
            for row, prompt, answer in zip(batch, prompts, answers):
                support = predict_support(row["dataset"], str(row["question"]), row["reader_context_docs"], sources[row["dataset"]][row["query_id"]])
                payload = {"dataset": row["dataset"], "query_id": row["query_id"], "method": row["method"], "reader": args.reader,
                           "predicted_answer": answer.strip(), "predicted_support": sorted(support),
                           "input_context_hash": row["context_hash"], "reader_input_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                           "reader_output_hash": hashlib.sha256(answer.strip().encode()).hexdigest(), "labels_loaded": False, "metrics_computed": False}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        completed = len(existing) + start + len(batch)
        if completed % 100 < len(batch):
            print(json.dumps({"reader": args.reader, "completed": completed, "total": len(contexts), "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    if sum(1 for _ in rows(args.output)) != 3600:
        raise ValueError("incomplete reader output")
    args.output.with_suffix(".completed.json").write_text(json.dumps({"status": "complete_unscored", "reader": args.reader, "rows": 3600, "labels_loaded": False, "metrics_computed": False, "contexts_sha256": sha256(args.contexts), "prediction_sha256": sha256(args.output)}, indent=2) + "\n")
    print(json.dumps({"status": "complete_unscored", "reader": args.reader, "rows": 3600}, indent=2))


if __name__ == "__main__":
    main()
