#!/usr/bin/env python3
"""Run a frozen Reader on baseline contexts without opening test labels."""
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


MODELS = {
    "flan": (
        "google/flan-t5-large",
        "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
    ),
    "unifiedqa": (
        "allenai/unifiedqa-v2-t5-large-1363200",
        "1d3b8e13b29dbd161494b0b15428378f4713c418",
    ),
}
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    context = " ".join(
        f"{document['title']}: {document['text']}" for document in documents
    )
    return f"{question} \n {context[:4000]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=tuple(MODELS), required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--v16-eval", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.v16_eval))
    from eval_common import normalize_title, source_documents, unit_features

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    contexts = list(rows(args.contexts))
    if not contexts:
        raise ValueError("contexts are empty")
    context_keys = {
        (row["dataset"], row["method"], row["query_id"]) for row in contexts
    }
    if len(context_keys) != len(contexts):
        raise ValueError("duplicate context keys")
    if any(row.get("gold_or_answer_used") for row in contexts):
        raise ValueError("label firewall violation in contexts")

    sources = {
        dataset: {
            query_id(row): row
            for row in rows(
                args.sample_root / f"{dataset}_final_test_inputs_n300.jsonl"
            )
        }
        for dataset in DATASETS
    }
    existing = list(rows(args.output)) if args.resume and args.output.exists() else []
    done = {
        (row["dataset"], row["method"], row["query_id"]) for row in existing
    }
    if len(done) != len(existing) or not done.issubset(context_keys):
        raise ValueError("invalid or duplicated resume output")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    pending = [
        row
        for row in contexts
        if (row["dataset"], row["method"], row["query_id"]) not in done
    ]
    pending_by_context: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in pending:
        content_key = (row["dataset"], row["query_id"], row["context_hash"])
        pending_by_context.setdefault(content_key, []).append(row)
    pending_groups = list(pending_by_context.values())

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
    ).to(args.device)
    model.eval()
    checkpoints = {
        dataset: joblib.load(args.v16_eval / f"checkpoints/{dataset}_support.joblib")
        for dataset in DATASETS
    }
    prompt_function = flan_prompt if args.reader == "flan" else unifiedqa_prompt

    def predict_support(
        dataset: str,
        question: str,
        documents: list[dict[str, str]],
        source_row: dict[str, Any],
    ) -> list[list[Any]]:
        local = {
            document["doc_id"]: document
            for document in source_documents(source_row, dataset)
        }
        instances = []
        for document_rank, document in enumerate(documents):
            exact = local.get(document["doc_id"])
            if dataset == "musique":
                identity = (
                    ("paragraph", int(exact["paragraph_idx"]))
                    if exact
                    else (
                        "document",
                        int(
                            hashlib.sha1(document["doc_id"].encode()).hexdigest()[:8],
                            16,
                        ),
                    )
                )
                instances.append(
                    {
                        "identity": identity,
                        "features": unit_features(
                            question,
                            document["title"],
                            document["text"],
                            document_rank,
                            0,
                            1,
                        ),
                    }
                )
                continue
            sentences = (
                exact.get("sentences", [])
                if exact
                else [
                    value.strip()
                    for value in re.split(r"(?<=[.!?])\s+", str(document["text"]))
                    if value.strip()
                ]
            )
            sentences = sentences or [str(document["text"])]
            for sentence_id, sentence in enumerate(sentences):
                instances.append(
                    {
                        "identity": (normalize_title(document["title"]), sentence_id),
                        "features": unit_features(
                            question,
                            document["title"],
                            sentence,
                            document_rank,
                            sentence_id,
                            len(sentences),
                        ),
                    }
                )
        probabilities = checkpoints[dataset]["model"].predict_proba(
            np.asarray([row["features"] for row in instances])
        )[:, 1]
        ranked = sorted(
            zip(instances, probabilities), key=lambda pair: pair[1], reverse=True
        )
        selected = [
            row
            for row, score in ranked
            if score >= checkpoints[dataset]["threshold"]
        ][:6]
        minimum = int(checkpoints[dataset].get("minimum_predictions", 2))
        if len(selected) < minimum:
            selected = [row for row, _ in ranked[:minimum]]
        return [list(row["identity"]) for row in selected]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = len(existing)
    for start in range(0, len(pending_groups), args.batch_size):
        groups = pending_groups[start : start + args.batch_size]
        batch = [group[0] for group in groups]
        prompts = [
            prompt_function(str(row["question"]), row["reader_context_docs"])
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=32, num_beams=1, do_sample=False
            )
        answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
        with args.output.open("a", encoding="utf-8") as handle:
            for aliases, row, prompt, answer in zip(groups, batch, prompts, answers):
                support = predict_support(
                    row["dataset"],
                    str(row["question"]),
                    row["reader_context_docs"],
                    sources[row["dataset"]][row["query_id"]],
                )
                for alias in aliases:
                    payload = {
                        "dataset": alias["dataset"],
                        "query_id": alias["query_id"],
                        "method": alias["method"],
                        "reader": args.reader,
                        "predicted_answer": answer.strip(),
                        "predicted_support": sorted(support),
                        "input_context_hash": alias["context_hash"],
                        "reader_input_hash": hashlib.sha256(
                            prompt.encode()
                        ).hexdigest(),
                        "reader_output_hash": hashlib.sha256(
                            answer.strip().encode()
                        ).hexdigest(),
                        "labels_loaded": False,
                        "metrics_computed": False,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                completed += len(aliases)
        if start % 100 < args.batch_size:
            print(
                json.dumps(
                    {
                        "reader": args.reader,
                        "completed": completed,
                        "total": len(contexts),
                        "unique_contexts_completed": min(
                            start + len(batch), len(pending_groups)
                        ),
                        "unique_contexts_total": len(pending_groups),
                        "elapsed_s": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )

    if sum(1 for _ in rows(args.output)) != len(contexts):
        raise ValueError("incomplete reader output")
    marker = {
        "status": "complete_unscored",
        "reader": args.reader,
        "repository": repository,
        "revision": revision,
        "rows": len(contexts),
        "unique_contexts_evaluated": len(pending_groups),
        "labels_loaded": False,
        "metrics_computed": False,
        "contexts_sha256": sha256(args.contexts),
        "prediction_sha256": sha256(args.output),
    }
    args.output.with_suffix(".completed.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
