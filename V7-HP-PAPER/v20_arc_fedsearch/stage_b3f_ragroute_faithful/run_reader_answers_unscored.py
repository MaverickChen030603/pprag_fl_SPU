#!/usr/bin/env python3
"""Run frozen Readers from materialized contexts without loading evaluation labels.

This B3F variant deliberately emits answer predictions only.  Unlike the
submission reader helper, it never opens the public source rows (which contain
answers and supporting annotations) merely to construct auxiliary support
predictions during the blind phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable


MODELS = {
    "flan": ("google/flan-t5-large", "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"),
    "unifiedqa": (
        "allenai/unifiedqa-v2-t5-large-1363200",
        "1d3b8e13b29dbd161494b0b15428378f4713c418",
    ),
}


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def flan_prompt(question: str, documents: list[dict[str, str]]) -> str:
    context = "\n".join(
        f"[{index}] {document['title']}: {document['text']}"
        for index, document in enumerate(documents, 1)
    )
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:4000]}\n\nAnswer:"
    )


def unifiedqa_prompt(question: str, documents: list[dict[str, str]]) -> str:
    context = " ".join(f"{row['title']}: {row['text']}" for row in documents)
    return f"{question} \n {context[:4000]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=tuple(MODELS), required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    contexts = list(rows(args.contexts))
    keys = {(row["dataset"], row["method"], row["query_id"]) for row in contexts}
    if len(contexts) != 3000 or len(keys) != len(contexts):
        raise ValueError("expected exactly 3,000 unique frozen B3F contexts")
    if any(row.get("gold_or_answer_used") or row.get("labels_loaded") for row in contexts):
        raise ValueError("label firewall violation in contexts")
    if args.output.exists():
        raise FileExistsError(args.output)

    repository, revision = MODELS[args.reader]
    tokenizer = AutoTokenizer.from_pretrained(
        repository,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=True,
        use_fast=args.reader == "flan",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repository,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(args.device).eval()
    prompt = flan_prompt if args.reader == "flan" else unifiedqa_prompt
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open("x", encoding="utf-8") as handle:
        for start in range(0, len(contexts), args.batch_size):
            batch = contexts[start : start + args.batch_size]
            prompts = [prompt(str(row["question"]), row["reader_context_docs"]) for row in batch]
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(args.device)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens, max_new_tokens=32, num_beams=1, do_sample=False
                )
            answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, text, answer in zip(batch, prompts, answers):
                payload = {
                    "dataset": row["dataset"],
                    "query_id": row["query_id"],
                    "method": row["method"],
                    "reader": args.reader,
                    "predicted_answer": answer.strip(),
                    "input_context_hash": row["context_hash"],
                    "reader_input_hash": hashlib.sha256(text.encode()).hexdigest(),
                    "reader_output_hash": hashlib.sha256(answer.strip().encode()).hexdigest(),
                    "labels_loaded": False,
                    "metrics_computed": False,
                    "predicted_support_available": False,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if start % 100 < args.batch_size:
                print(
                    json.dumps(
                        {
                            "reader": args.reader,
                            "completed": min(start + len(batch), len(contexts)),
                            "total": len(contexts),
                            "elapsed_s": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
