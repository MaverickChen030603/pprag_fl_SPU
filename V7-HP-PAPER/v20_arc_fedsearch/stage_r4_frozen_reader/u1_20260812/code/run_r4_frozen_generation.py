#!/usr/bin/env python3
"""Gold-blind Hotpot-only R4 generation with per-cell checksum resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

PROMPT_LITERAL_FLAN = 'f"Answer the question using only the context. Return a short answer.\\n\\nQuestion: {question}\\n\\nContext:\\n{context[:max_chars]}\\n\\nAnswer:"'
PROMPT_LITERAL_UNIFIEDQA = 'f"{question} \\n {context[:max_chars]}"'
READERS = {
    "flan_t5_large": {
        "model_id": "google/flan-t5-large",
        "revision": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
        "use_fast": True,
    },
    "unifiedqa_t5_large": {
        "model_id": "allenai/unifiedqa-v2-t5-large-1363200",
        "revision": "1d3b8e13b29dbd161494b0b15428378f4713c418",
        "use_fast": False,
    },
}
METHODS = ("inherited", "label_free", "logistic", "centralized")
REQUIRED_RECORD_FIELDS = {
    "query_id", "reader", "method", "prediction", "generation_metadata",
    "input_hash", "protocol_hash", "model_revision", "tokenizer_revision",
    "context_checksum", "record_checksum",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{number}: {exc}") from exc
    return rows


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def prompt_flan(question: str, docs: list[dict], max_chars: int) -> str:
    context = "\n".join(f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, 1))
    return f"Answer the question using only the context. Return a short answer.\n\nQuestion: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"


def prompt_unifiedqa(question: str, docs: list[dict], max_chars: int) -> str:
    context = " ".join(f"{doc['title']}: {doc['text']}" for doc in docs)
    return f"{question} \n {context[:max_chars]}"


def validate_inputs(query_view: Path, contexts_path: Path, method: str) -> tuple[list[dict], list[dict]]:
    queries = read_jsonl(query_view)
    contexts = read_jsonl(contexts_path)
    if len(queries) != 300 or len(contexts) != 300:
        raise ValueError("canonical generation requires exactly 300 queries and 300 contexts")
    if any(set(row) != {"query_id", "question"} for row in queries):
        raise ValueError("query view schema must be exactly query_id + question")
    qids = [str(row["query_id"]) for row in queries]
    context_ids = [str(row["query_id"]) for row in contexts]
    if len(set(qids)) != 300 or qids != context_ids:
        raise ValueError("query/context ID set or order mismatch")
    if any(row.get("method") != method for row in contexts):
        raise ValueError("context method mismatch")
    if any(row.get("gold_or_answer_used") is not False for row in contexts):
        raise ValueError("context is not marked gold-blind")
    if any(len(row.get("reader_context_docs", [])) != 5 for row in contexts):
        raise ValueError("reader context document count must be 5")
    return queries, contexts


def validate_existing(path: Path, expected: dict, expected_ids: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    output = {}
    for row in read_jsonl(path):
        if set(row) != REQUIRED_RECORD_FIELDS:
            raise ValueError("existing prediction record schema mismatch")
        query_id = str(row["query_id"])
        if query_id not in expected_ids or query_id in output:
            raise ValueError("existing prediction has unexpected or duplicate query ID")
        payload = {key: value for key, value in row.items() if key != "record_checksum"}
        if canonical_hash(payload) != row["record_checksum"]:
            raise ValueError(f"record checksum mismatch: {query_id}")
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"existing prediction {query_id} mismatches {key}")
        output[query_id] = row
    return output


def seal_cell(output: Path, records: dict[str, dict], query_ids: list[str], seal: dict) -> dict:
    if set(records) != set(query_ids) or len(records) != len(query_ids):
        raise ValueError("cannot seal incomplete cell")
    ordered = [records[query_id] for query_id in query_ids]
    temp = output.with_name(output.name + f".seal.{os.getpid()}")
    with temp.open("x", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, output)
    output.chmod(0o444)
    completed = output.with_suffix(".completed_query_ids.json")
    marker = output.with_suffix(".completed.json")
    atomic_json(completed, {
        "n": len(query_ids),
        "query_ids": query_ids,
        "query_id_order_sha256": digest_lines(query_ids),
    })
    marker_payload = {
        **seal,
        "status": "complete",
        "n": len(query_ids),
        "prediction_file": str(output),
        "prediction_file_sha256": sha256(output),
        "completed_query_ids_manifest": str(completed),
        "completed_query_ids_manifest_sha256": sha256(completed),
    }
    atomic_json(marker, marker_payload)
    completed.chmod(0o444)
    marker.chmod(0o444)
    return marker_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=tuple(READERS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--query-view", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    model = READERS[args.reader]
    frozen = protocol["readers"][args.reader]
    if frozen["model_revision"] != model["revision"] or frozen["tokenizer_revision"] != model["revision"]:
        raise ValueError("runner/model revision differs from frozen protocol")
    if protocol["seed"] != 20260812 or protocol["batch_size"] != 4:
        raise ValueError("frozen seed/batch size mismatch")
    queries, contexts = validate_inputs(args.query_view, args.contexts, args.method)
    query_ids = [str(row["query_id"]) for row in queries]
    context_checksum = sha256(args.contexts)
    output = args.output_root / args.reader / f"{args.method}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_manifest = output.with_suffix(".expected_query_ids.json")
    expected_payload = {"n": 300, "query_ids": query_ids, "query_id_order_sha256": digest_lines(query_ids)}
    if expected_manifest.exists():
        if json.loads(expected_manifest.read_text()) != expected_payload:
            raise ValueError("existing expected-query manifest mismatch")
    else:
        atomic_json(expected_manifest, expected_payload)
        expected_manifest.chmod(0o444)
    expected = {
        "reader": args.reader,
        "method": args.method,
        "protocol_hash": protocol_hash,
        "model_revision": model["revision"],
        "tokenizer_revision": model["revision"],
        "context_checksum": context_checksum,
    }
    records = validate_existing(output, expected, set(query_ids))
    marker = output.with_suffix(".completed.json")
    if marker.exists():
        marker_payload = json.loads(marker.read_text())
        if len(records) == 300 and marker_payload.get("prediction_file_sha256") == sha256(output):
            print(json.dumps({"status": "cache_hit_complete", "records": 300, "cell": [args.reader, args.method]}))
            return
        raise ValueError("completed marker/checksum mismatch")
    pending = [(query, context) for query, context in zip(queries, contexts) if str(query["query_id"]) not in records]
    if not pending:
        seal_cell(output, records, query_ids, {**expected, "expected_manifest_sha256": sha256(expected_manifest)})
        return

    import numpy as np
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        model["model_id"], revision=model["revision"], local_files_only=True, use_fast=model["use_fast"]
    )
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    reader_model = AutoModelForSeq2SeqLM.from_pretrained(
        model["model_id"], revision=model["revision"], local_files_only=True, torch_dtype=dtype
    ).to(torch.device(args.device))
    reader_model.eval()
    prompt_fn = prompt_flan if args.reader == "flan_t5_large" else prompt_unifiedqa
    max_chars = protocol["max_context_chars"]
    started = time.perf_counter()
    output.chmod(0o644) if output.exists() else None
    with output.open("a", encoding="utf-8") as f:
        for start in range(0, len(pending), protocol["batch_size"]):
            batch = pending[start:start + protocol["batch_size"]]
            prompts = [prompt_fn(str(query["question"]), context["reader_context_docs"], max_chars) for query, context in batch]
            encoded = tokenizer(
                prompts, padding=True, truncation=True, max_length=protocol["max_source_tokens"], return_tensors="pt"
            ).to(args.device)
            call_started = time.perf_counter()
            with torch.inference_mode():
                generated = reader_model.generate(
                    **encoded,
                    max_new_tokens=protocol["max_new_tokens"],
                    num_beams=protocol["num_beams"],
                    do_sample=protocol["do_sample"],
                )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            per_item_seconds = (time.perf_counter() - call_started) / len(batch)
            for (query, context), prompt, prediction in zip(batch, prompts, predictions):
                query_id = str(query["query_id"])
                payload = {
                    "query_id": query_id,
                    **expected,
                    "prediction": prediction.strip(),
                    "generation_metadata": {
                        "input_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "runtime_seconds": per_item_seconds,
                        "seed": protocol["seed"],
                        "max_source_tokens": protocol["max_source_tokens"],
                        "max_new_tokens": protocol["max_new_tokens"],
                        "num_beams": protocol["num_beams"],
                        "do_sample": protocol["do_sample"],
                        "dtype": "float16" if args.device.startswith("cuda") else "float32",
                        "device": args.device,
                    },
                    "input_hash": canonical_hash({
                        "query_id": query_id,
                        "question": query["question"],
                        "context_sha256": context["context_sha256"],
                        "reader": args.reader,
                    }),
                }
                payload["record_checksum"] = canonical_hash(payload)
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                records[query_id] = payload
            f.flush()
            os.fsync(f.fileno())
            print(json.dumps({
                "cell": [args.reader, args.method],
                "completed": len(records),
                "expected": 300,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            }), flush=True)
    seal = seal_cell(output, records, query_ids, {**expected, "expected_manifest_sha256": sha256(expected_manifest)})
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
