#!/usr/bin/env python3
"""Frozen Stage-B FLAN generation with no evaluator or gold import."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np


MODEL_ID = "google/flan-t5-large"
REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
EXPECTED_QUERIES = 4200
EXPECTED_ROWS = 8400
METHODS = ("federated_baseline", "logistic_proberoute")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def words(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", str(value).lower()) if len(token) > 1]


def unit_features(question: str, title: str, text: str, doc_rank: int, unit_rank: int, unit_count: int):
    question_tokens, text_tokens, title_tokens = set(words(question)), set(words(text)), set(words(title))
    union = question_tokens | text_tokens
    capitals_q = set(re.findall(r"\b[A-Z][A-Za-z0-9'-]+", question))
    capitals_d = set(re.findall(r"\b[A-Z][A-Za-z0-9'-]+", f"{title} {text}"))
    return [
        len(question_tokens & text_tokens) / len(question_tokens) if question_tokens else 0.0,
        len(question_tokens & title_tokens) / len(question_tokens) if question_tokens else 0.0,
        len(question_tokens & text_tokens) / len(union) if union else 0.0,
        len(capitals_q & capitals_d) / len(capitals_q | capitals_d) if capitals_q and capitals_d else 0.0,
        float(unit_rank == 0), unit_rank / max(1, unit_count - 1), min(doc_rank, 4) / 4.0,
        min(len(text_tokens), 120) / 120.0,
        len(title_tokens & text_tokens) / len(title_tokens) if title_tokens else 0.0,
    ]


def support_prediction(checkpoint, question: str, docs: list[dict]):
    instances = []
    for doc_rank, doc in enumerate(docs):
        sentences = [value.strip() for value in re.split(r"(?<=[.!?])\s+", str(doc["text"])) if value.strip()]
        sentences = sentences or [str(doc["text"])]
        for sentence_id, sentence in enumerate(sentences):
            instances.append(
                {
                    "identity": [normalize_title(doc["title"]), sentence_id],
                    "features": unit_features(question, doc["title"], sentence, doc_rank, sentence_id, len(sentences)),
                }
            )
    probability = checkpoint["model"].predict_proba(np.asarray([row["features"] for row in instances]))[:, 1]
    ranked = sorted(zip(instances, probability), key=lambda pair: pair[1], reverse=True)
    selected = [row for row, score in ranked if score >= checkpoint["threshold"]][:6]
    minimum = int(checkpoint.get("minimum_predictions", 2))
    if len(selected) < minimum:
        selected = [row for row, _ in ranked[:minimum]]
    return sorted(row["identity"] for row in selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 4:
        raise ValueError("batch size is frozen at 4")
    split_rows = list(rows(args.split))
    if len(split_rows) != EXPECTED_QUERIES or any(set(row) != {"query_id", "question"} for row in split_rows):
        raise ValueError("generation source is not the frozen question-only view")
    contexts = list(rows(args.contexts))
    if len(contexts) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} contexts")
    expected_keys = {(str(row["query_id"]), method) for row in split_rows for method in METHODS}
    context_keys = {(str(row["query_id"]), str(row["method"])) for row in contexts}
    if context_keys != expected_keys:
        raise ValueError("context primary-key mismatch")

    contract_path = args.output.with_suffix(".resume_contract.json")
    contract = {
        "contexts_sha256": sha256(args.contexts),
        "split_sha256": sha256(args.split),
        "support_checkpoint_sha256": sha256(args.support_checkpoint),
        "model_id": MODEL_ID,
        "revision": REVISION,
    }
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise RuntimeError("resume contract mismatch")
    else:
        contract_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    existing = list(rows(args.output)) if args.output.exists() else []
    done = {(str(row["query_id"]), str(row["method"])) for row in existing}
    if len(done) != len(existing) or not done <= expected_keys:
        raise ValueError("duplicate or foreign resume row")
    pending = [row for row in contexts if (str(row["query_id"]), str(row["method"])) not in done]

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    checkpoint = joblib.load(args.support_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = []
        for row in batch:
            context = "\n".join(
                f"[{index}] {doc['title']}: {doc['text']}"
                for index, doc in enumerate(row["reader_context_docs"], 1)
            )
            prompts.append(
                f"Answer the question using only the context. Return a short answer.\n\n"
                f"Question: {row['question']}\n\nContext:\n{context[:4000]}\n\nAnswer:"
            )
        encoded = tokenizer(
            prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        answers = tokenizer.batch_decode(generated, skip_special_tokens=True)
        with args.output.open("a", encoding="utf-8") as handle:
            for row, answer in zip(batch, answers):
                payload = {
                    "dataset": "hotpotqa",
                    "query_id": str(row["query_id"]),
                    "method": str(row["method"]),
                    "reader": "flan_t5_large",
                    "predicted_answer": answer.strip(),
                    "predicted_support": support_prediction(
                        checkpoint, str(row["question"]), list(row["reader_context_docs"])
                    ),
                    "context_hash": str(row.get("context_hash", "")),
                    "generation_read_gold": False,
                    "evaluator_imported": False,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    final_rows = list(rows(args.output))
    final_keys = {(str(row["query_id"]), str(row["method"])) for row in final_rows}
    if len(final_rows) != EXPECTED_ROWS or final_keys != expected_keys:
        raise RuntimeError("generation incomplete")
    complete = args.output.with_suffix(".complete.json")
    temp = complete.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {
                "status": "stage_b_predictions_complete_unscored",
                "rows": EXPECTED_ROWS,
                "output_sha256": sha256(args.output),
                "rerun_forbidden": True,
                "gold_accessed": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    os.replace(temp, complete)


if __name__ == "__main__":
    import os

    main()
