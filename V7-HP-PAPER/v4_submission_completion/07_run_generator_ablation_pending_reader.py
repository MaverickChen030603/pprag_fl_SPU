#!/usr/bin/env python3
"""Evaluate only generator-ablation contexts not covered by prior reader runs."""

from __future__ import annotations

import argparse
import json
import time

from completion_common import ABLATION, V4_ROOT, add_v4_import_path, ensure_layout, read_jsonl, write_json, write_jsonl


FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"


def prompt(question: str, docs: list[dict], max_chars: int = 3200) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, start=1))
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"
    )


def merge(num_shards: int) -> None:
    rows = []
    for shard_id in range(num_shards):
        path = ABLATION / "reader" / f"pending_outcomes_shard_{shard_id:02d}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    pending = read_jsonl(ABLATION / "pending_context_actions.jsonl")
    if len(rows) != len(pending):
        raise AssertionError(f"Pending outcome rows {len(rows)} != contexts {len(pending)}")
    write_jsonl(ABLATION / "reader" / "pending_context_outcomes.jsonl", rows)
    write_json(ABLATION / "reader" / "summary.json", {"status": "complete", "n_rows": len(rows)})
    print(json.dumps({"status": "complete", "rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    ensure_layout()
    if args.merge_only:
        merge(args.num_shards)
        return
    add_v4_import_path()
    from v4_common import answer_scores, load_source_examples, title_metrics

    pending = read_jsonl(ABLATION / "pending_context_actions.jsonl")
    shard = pending[args.shard_id :: args.num_shards]
    source = load_source_examples()
    output_dir = ABLATION / "reader"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pending_outcomes_shard_{args.shard_id:02d}.jsonl"
    existing = read_jsonl(output_path) if args.resume and output_path.exists() else []
    done = {str(row["context_signature"]) for row in existing}
    if output_path.exists() and not args.resume:
        output_path.unlink()
    queue = [row for row in shard if str(row["context_signature"]) not in done]

    if queue:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(FLAN, local_files_only=True, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(FLAN, local_files_only=True, torch_dtype=torch.float16).to(args.device)
        model.eval()
        for start in range(0, len(queue), args.batch_size):
            batch = queue[start : start + args.batch_size]
            encoded = tokenizer(
                [prompt(row["question"], row["context_docs"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(args.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            rows = []
            for action, prediction in zip(batch, predictions):
                item = source[str(action["query_id"])]
                _, answer_f1 = answer_scores(prediction.strip(), item["answer"])
                title_recall, title_f1 = title_metrics(action["context_titles"], item["supporting_titles"])
                rows.append({
                    "query_id": str(action["query_id"]),
                    "context_signature": str(action["context_signature"]),
                    "context_doc_ids": list(action["context_doc_ids"]),
                    "prediction": prediction.strip(),
                    "answer_f1": answer_f1,
                    "title_recall": title_recall,
                    "title_f1": title_f1,
                    "answer_title_product": answer_f1 * title_f1,
                    "source": "ablation_new_reader_run",
                })
            write_jsonl(output_path, rows, mode="a")
            completed = len(existing) + start + len(batch)
            if completed % 256 < args.batch_size:
                write_json(output_dir / f"progress_shard_{args.shard_id:02d}.json", {
                    "completed": completed,
                    "total": len(shard),
                    "epoch": time.time(),
                })
    print(json.dumps({"status": "shard_complete", "shard": args.shard_id, "rows": len(shard)}, indent=2))


if __name__ == "__main__":
    main()
