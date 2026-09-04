#!/usr/bin/env python3
"""Run the unchanged FLAN reader over every frozen-transfer 2Wiki action."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from completion_common import EXTERNAL, V4_ROOT, add_v4_import_path, ensure_layout, read_json, read_jsonl, write_json, write_jsonl


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
        path = EXTERNAL / "reader" / f"action_outcomes_shard_{shard_id:02d}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    actions = read_jsonl(EXTERNAL / "generated_actions_1000.jsonl")
    if len(rows) != len(actions):
        raise AssertionError(f"Reader output rows {len(rows)} != action rows {len(actions)}")
    if len({str(row['action_id']) for row in rows}) != len(rows):
        raise AssertionError("Duplicate action IDs in merged reader output")
    rows.sort(key=lambda row: str(row["action_id"]))
    output = EXTERNAL / "reader" / "all_action_outcomes.jsonl"
    write_jsonl(output, rows)
    write_json(EXTERNAL / "reader" / "summary.json", {
        "status": "complete",
        "reader": "google/flan-t5-large",
        "reader_revision": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
        "n_queries": len({str(row["query_id"]) for row in rows}),
        "n_action_rows": len(rows),
        "prompt_or_decoding_retuned": False,
        "generation": {"max_new_tokens": 32, "num_beams": 1, "do_sample": False},
    })
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
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard")
    add_v4_import_path()
    from v4_common import answer_access, answer_scores, title_metrics

    audit = read_json(EXTERNAL / "frozen_generator_selector_audit.json")
    if audit.get("status") != "pass" or audit.get("target_tuning"):
        raise AssertionError("Frozen transfer audit must pass before reader inference")
    source = {str(row["query_id"]): row for row in read_json(EXTERNAL / "2wiki_frozen_1000.json")}
    all_actions = read_jsonl(EXTERNAL / "generated_actions_1000.jsonl")
    query_ids = sorted(source)
    shard_ids = set(query_ids[args.shard_id :: args.num_shards])
    actions = [row for row in all_actions if str(row["query_id"]) in shard_ids]
    output_dir = EXTERNAL / "reader"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"action_outcomes_shard_{args.shard_id:02d}.jsonl"
    existing = read_jsonl(output_path) if args.resume and output_path.exists() else []
    done = {str(row["action_id"]) for row in existing}
    if output_path.exists() and not args.resume:
        output_path.unlink()
    pending = [row for row in actions if str(row["action_id"]) not in done]

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    runtime = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(FLAN, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN, local_files_only=True, torch_dtype=torch.float16).to(runtime)
    model.eval()
    write_json(output_dir / f"environment_shard_{args.shard_id:02d}.json", {
        "reader": "google/flan-t5-large",
        "model_path": FLAN,
        "device": args.device,
        "batch_size": args.batch_size,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "n_queries": len(shard_ids),
        "n_actions": len(actions),
        "prompt_unchanged_from_hotpot_v4": True,
        "decoding_retuned": False,
    })
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [prompt(str(row["question"]), list(row["context_docs"])) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(runtime)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        output_rows = []
        for action, prediction in zip(batch, predictions):
            item = source[str(action["query_id"])]
            answer_em, answer_f1 = answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = title_metrics(action["context_titles"], item["supporting_titles"])
            output_rows.append({
                "query_id": str(action["query_id"]),
                "action_id": str(action["action_id"]),
                "outer_fold": int(action["outer_fold"]),
                "action_family": str(action["action_family"]),
                "prediction": prediction.strip(),
                "answer_em": answer_em,
                "answer_f1": answer_f1,
                "answer_access_at_5": answer_access(item["answer"], action["context_docs"]),
                "title_recall": title_recall,
                "title_f1": title_f1,
                "answer_title_product": answer_f1 * title_f1,
                "context_titles": list(action["context_titles"]),
            })
        write_jsonl(output_path, output_rows, mode="a")
        completed = len(existing) + start + len(batch)
        if completed % 256 < args.batch_size:
            write_json(output_dir / f"progress_shard_{args.shard_id:02d}.json", {
                "completed": completed,
                "total": len(actions),
                "epoch": time.time(),
            })
    print(json.dumps({"status": "shard_complete", "shard": args.shard_id, "rows": len(actions)}, indent=2))


if __name__ == "__main__":
    main()
