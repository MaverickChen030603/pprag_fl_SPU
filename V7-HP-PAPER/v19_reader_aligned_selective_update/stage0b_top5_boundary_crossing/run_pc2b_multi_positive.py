#!/usr/bin/env python3
"""PC-2B: multi-positive support-aware loss with boundary pairwise term."""

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


def build_examples(development: Path, frozen_pool: Path, output: Path, per_query_negatives: int = 4) -> list[dict[str, Any]]:
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
        support_docs = sorted(support_docs, key=lambda item: item[0], reverse=True)
        boundary = [
            doc for idx, doc in enumerate(docs, start=1)
            if 4 <= idx <= 10 and norm(doc["title"]) not in gold
        ]
        fallback = [
            doc for idx, doc in enumerate(docs, start=1)
            if idx > 10 and norm(doc["title"]) not in gold
        ]
        negatives = (boundary + fallback)[:per_query_negatives]
        if len(negatives) < per_query_negatives:
            continue
        examples.append(
            {
                "query_id": qid,
                "query": dev[qid]["question"],
                "positives": [doc_text(doc) for _, doc in support_docs],
                "positive_ranks": [rank for rank, _ in support_docs],
                "negatives": [doc_text(doc) for doc in negatives],
                "negative_ranks": [
                    next(idx for idx, candidate in enumerate(docs, start=1) if candidate["doc_id"] == doc["doc_id"])
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
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--lambda-boundary", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_examples(args.development, args.frozen_pool, args.output_dir / "multi_positive_training_manifest.jsonl")
    if not data:
        raise RuntimeError("No PC-2B multi-positive examples were available.")
    (args.output_dir / "config.yaml").write_text(
        "\n".join(
            [
                "variant: PC-2B",
                "single_changed_factor: multi_positive_support_aware_loss",
                f"steps: {args.steps}",
                f"batch_size: {args.batch_size}",
                f"learning_rate: {args.lr}",
                f"pairwise_margin: {args.margin}",
                f"lambda_boundary: {args.lambda_boundary}",
                "lora_rank: 8",
                "payload: unchanged_from_pc1",
                "reader: disabled",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

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
        pos_masks: list[list[bool]] = []
        neg_masks: list[list[bool]] = []
        max_docs = max(len(item["positives"]) + len(item["negatives"]) for item in batch)
        for item in batch:
            local_docs = item["positives"] + item["negatives"]
            docs.extend(local_docs + [local_docs[-1]] * (max_docs - len(local_docs)))
            pos_masks.append([True] * len(item["positives"]) + [False] * (max_docs - len(item["positives"])))
            neg_masks.append([False] * len(item["positives"]) + [True] * len(item["negatives"]) + [False] * (max_docs - len(local_docs)))
        doc_batch = tokenizer(docs, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        model.train()
        query_embeddings = pooled(model, query_batch)
        doc_embeddings = pooled(model, doc_batch).reshape(len(batch), max_docs, -1)
        logits = (query_embeddings.unsqueeze(1) * doc_embeddings).sum(-1) * 20
        pos_mask = torch.tensor(pos_masks, device=device)
        neg_mask = torch.tensor(neg_masks, device=device)
        pos_logits = logits.masked_fill(~pos_mask, -1e9)
        all_valid = pos_mask | neg_mask
        valid_logits = logits.masked_fill(~all_valid, -1e9)
        multi_positive_loss = -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(valid_logits, dim=1)).mean()
        pairwise = torch.relu(args.margin - logits.unsqueeze(2) + logits.unsqueeze(1))
        pairwise_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        boundary_loss = pairwise[pairwise_mask].mean()
        loss = multi_positive_loss + args.lambda_boundary * boundary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 1.0))
        optimizer.step()
        logs.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "multi_positive_loss": float(multi_positive_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
                "gradient_norm": grad_norm,
            }
        )

    state = adapter_state(model)
    torch.save({"adapter_state": state, "method": "centralized_pc2b_multi_positive", "seed": args.seed}, args.output_dir / "adapter.pt")
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
        "control": "PC-2B",
        "single_changed_factor": "multi_positive_support_aware_loss",
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
