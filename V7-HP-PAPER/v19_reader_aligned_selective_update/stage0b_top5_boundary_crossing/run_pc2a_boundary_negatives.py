#!/usr/bin/env python3
"""PC-2A: boundary hard-negative curriculum with PC-1 training strength."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

V19 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V19 / "stage0_full_upload"))
sys.path.insert(0, str(V19 / "model"))
from lora_blocks import adapter_state, load_adapter_state, state_bytes
from run_stage0_viability import evaluate_pool, make_model, pooled


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(text: Any) -> str:
    return " ".join(str(text).lower().split())


def support_titles(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", {})
    if isinstance(facts, dict):
        return {norm(x) for x in facts.get("title", [])}
    return {norm(x[0]) for x in facts}


def ranked(row: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(row["pool"], key=lambda d: (-float(d.get("hybrid_score", d.get("retrieval_score", 0.0))), str(d["doc_id"])))


def doc_text(doc: dict[str, Any]) -> str:
    return f"{doc['title']}: {doc.get('text', '')}"


def build_boundary_examples(development: Path, frozen_pool: Path, output: Path, per_query: int = 4) -> list[dict[str, Any]]:
    dev = {str(row.get("query_id", row.get("id"))): row for row in read_jsonl(development)}
    examples: list[dict[str, Any]] = []
    for pool_row in read_jsonl(frozen_pool):
        qid = str(pool_row["query_id"])
        if qid not in dev:
            continue
        gold = support_titles(dev[qid])
        docs = ranked(pool_row)
        support_docs = [(idx + 1, doc) for idx, doc in enumerate(docs) if norm(doc["title"]) in gold]
        if not support_docs:
            continue
        # Push the currently weakest visible support rather than an already-easy first hop.
        positive_rank, positive_doc = max(support_docs, key=lambda item: item[0])
        boundary = [
            doc for idx, doc in enumerate(docs, start=1)
            if 4 <= idx <= 10 and norm(doc["title"]) not in gold
        ]
        fallback = [
            doc for idx, doc in enumerate(docs, start=1)
            if idx > 10 and norm(doc["title"]) not in gold
        ]
        negatives = (boundary + fallback)[:per_query]
        if len(negatives) < per_query:
            continue
        examples.append(
            {
                "query_id": qid,
                "query": dev[qid]["question"],
                "positive": doc_text(positive_doc),
                "positive_rank": positive_rank,
                "all_support_ranks": [rank for rank, _ in support_docs],
                "negatives": [doc_text(doc) for doc in negatives],
                "negative_ranks": [
                    next(idx for idx, candidate in enumerate(docs, start=1) if candidate["doc_id"] == doc["doc_id"])
                    for doc in negatives
                ],
                "negative_provenance": [
                    "rank_4_10_boundary" if 4 <= next(idx for idx, candidate in enumerate(docs, start=1) if candidate["doc_id"] == doc["doc_id"]) <= 10 else "rank_gt10_fallback"
                    for doc in negatives
                ],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--frozen-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "boundary_training_manifest.jsonl"
    data = build_boundary_examples(args.development, args.frozen_pool, manifest_path)
    if not data:
        raise RuntimeError("No PC-2A boundary examples were available.")

    config_text = "\n".join(
        [
            "variant: PC-2A",
            "single_changed_factor: boundary_hard_negative_curriculum",
            f"steps: {args.steps}",
            f"batch_size: {args.batch_size}",
            f"learning_rate: {args.lr}",
            "lora_rank: 8",
            "lora_alpha: 16.0",
            "payload: unchanged_from_pc1",
            "reader: disabled",
            "calibration_final_labels: not_read",
        ]
    )
    (args.output_dir / "config.yaml").write_text(config_text + "\n", encoding="utf-8")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = make_model(args.model, device, 8, 16.0)
    initial = torch.load(args.frozen_checkpoint, map_location="cpu")["adapter_state"]
    load_adapter_state(model, initial, device)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr)
    logs: list[dict[str, float | int]] = []

    for step in range(args.steps):
        batch = [data[(step * args.batch_size + idx) % len(data)] for idx in range(args.batch_size)]
        query_batch = tokenizer(
            ["Represent this sentence for searching relevant passages: " + item["query"] for item in batch],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        docs: list[str] = []
        for item in batch:
            docs += [item["positive"], *item["negatives"]]
        doc_batch = tokenizer(docs, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        model.train()
        query_embeddings = pooled(model, query_batch)
        doc_embeddings = pooled(model, doc_batch).reshape(len(batch), 5, -1)
        logits = (query_embeddings.unsqueeze(1) * doc_embeddings).sum(-1) * 20
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 1.0))
        optimizer.step()
        logs.append({"step": step, "loss": float(loss.detach().cpu()), "gradient_norm": grad_norm})

    state = adapter_state(model)
    torch.save({"adapter_state": state, "method": "centralized_pc2a_boundary_negatives", "seed": args.seed}, args.output_dir / "adapter.pt")
    metrics = evaluate_pool(
        model,
        tokenizer,
        state,
        args.development,
        args.pool,
        "hotpotqa",
        args.output_dir / "adapted_pool.jsonl",
        args.output_dir / "contexts.jsonl",
        100,
        device,
        32,
    )
    with (args.output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)
    result = {
        "status": "complete",
        "control": "PC-2A",
        "single_changed_factor": "boundary_hard_negative_curriculum",
        "examples": len(data),
        "steps": args.steps,
        "lr": args.lr,
        "adapter_bytes": state_bytes(state),
        "loss_first": logs[0]["loss"],
        "loss_last": logs[-1]["loss"],
        **metrics,
    }
    (args.output_dir / "training_log.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
