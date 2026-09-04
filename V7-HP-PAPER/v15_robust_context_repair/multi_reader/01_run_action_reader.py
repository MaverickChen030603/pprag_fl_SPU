#!/usr/bin/env python3
"""Label frozen complete-context actions with one frozen reader and SP model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retrieval"))
sys.path.insert(0, str(ROOT / "evaluation"))
from retrieval_common import documents, iter_rows, normalize_title, query_id
from eval_common import gold_support, official_metrics, sentence_features


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flan_prompt(question, docs, max_chars=3200):
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, 1))
    return f"Answer the question using only the context. Return a short answer.\n\nQuestion: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"


def unifiedqa_prompt(question, docs, max_chars=3200):
    context = " ".join(f"{doc['title']}: {doc['text']}" for doc in docs)[:max_chars]
    return f"{question} \n {context}"


def sentences_for_doc(doc):
    values = doc.get("sentences")
    if values:
        return [str(value) for value in values]
    values = [value.strip() for value in re.split(r"(?<=[.!?])\s+", str(doc.get("text", ""))) if value.strip()]
    return values or [str(doc.get("text", ""))]


def predict_support(checkpoint, question, docs):
    instances = []
    for rank, doc in enumerate(docs):
        sentences = sentences_for_doc(doc)
        for sent_id, sentence in enumerate(sentences):
            instances.append({"title": doc["title"], "sent_id": sent_id, "features": sentence_features(question, doc["title"], sentence, rank, sent_id, len(sentences))})
    model, threshold = checkpoint["model"], checkpoint["threshold"]
    probability = model.predict_proba(np.asarray([row["features"] for row in instances]))[:, 1]
    ranked = sorted(zip(instances, probability), key=lambda pair: pair[1], reverse=True)
    selected = [row for row, score in ranked if score >= threshold][:5]
    if len(selected) < 2:
        selected = [row for row, _ in ranked[:2]]
    return {(normalize_title(row["title"]), int(row["sent_id"])) for row in selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=("flan", "unifiedqa"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa"), required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--support-predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-actions-per-query", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    source = {query_id(row): row for row in iter_rows(args.split)}
    pools = {str(row["query_id"]): row for row in read_jsonl(args.pool)}
    grouped_actions = defaultdict(list)
    for action in read_jsonl(args.actions):
        grouped_actions[str(action["query_id"])].append(action)
    query_ids = sorted(grouped_actions)
    if args.max_queries:
        query_ids = query_ids[:args.max_queries]
    specs = []
    for qid in query_ids:
        values = sorted(grouped_actions[qid], key=lambda row: (not bool(row.get("is_baseline")), -float(row.get("cheap_score", 0.0)), str(row["action_id"])))
        specs.extend(values[:args.max_actions_per_query])

    existing = read_jsonl(args.output) if args.resume and args.output.exists() else []
    done = {(str(row["query_id"]), str(row["action_id"])) for row in existing}
    pending = [row for row in specs if (str(row["query_id"]), str(row["action_id"])) not in done]
    if args.output.exists() and not args.resume:
        args.output.unlink()

    runtime = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=args.reader == "flan")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float16).to(runtime)
    model.eval()
    support_checkpoint = joblib.load(args.support_predictor)
    prompt_fn = flan_prompt if args.reader == "flan" else unifiedqa_prompt

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        contexts = []
        for action in batch:
            qid = str(action["query_id"])
            pool_docs = {doc["doc_id"]: dict(doc) for doc in pools[qid]["documents"]}
            for local in documents(source[qid], args.dataset):
                if local["doc_id"] in pool_docs:
                    pool_docs[local["doc_id"]]["sentences"] = local["sentences"]
            contexts.append([pool_docs[doc_id] for doc_id in action["doc_ids"]])
        prompts = [prompt_fn(str(source[str(action["query_id"])]["question"]), context) for action, context in zip(batch, contexts)]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(runtime)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        output_rows = []
        for action, context, prediction in zip(batch, contexts, predictions):
            qid = str(action["query_id"])
            item = source[qid]
            support = predict_support(support_checkpoint, str(item["question"]), context)
            metrics = official_metrics(prediction.strip(), str(item["answer"]), support, gold_support(item))
            output_rows.append({"query_id": qid, "action_id": action["action_id"], "reader": args.reader, "is_baseline": bool(action.get("is_baseline")), "prediction": prediction.strip(), "context_doc_ids": action["doc_ids"], "predicted_support": sorted([list(value) for value in support]), **metrics})
        write_jsonl(args.output, output_rows, mode="a")
        if (start + len(batch)) % max(120, args.batch_size) < args.batch_size:
            print(json.dumps({"reader": args.reader, "completed": len(existing) + start + len(batch), "total": len(specs)}), flush=True)

    rows = read_jsonl(args.output)
    by_query = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    final = []
    for qid, values in by_query.items():
        baseline = next(row for row in values if row["is_baseline"])
        for row in values:
            row.update({"answer_delta": row["answer_f1"] - baseline["answer_f1"], "sp_delta": row["sp_f1"] - baseline["sp_f1"], "joint_delta": row["joint_f1"] - baseline["joint_f1"], "answer_drop": int(row["answer_f1"] < baseline["answer_f1"] - 1e-12), "joint_drop": int(row["joint_f1"] < baseline["joint_f1"] - 1e-12)})
            final.append(row)
    write_jsonl(args.output, final)
    print(json.dumps({"status": "complete", "reader": args.reader, "queries": len(by_query), "rows": len(final), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

