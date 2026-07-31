#!/usr/bin/env python3
"""Stage 0: real adapter adaptation before any selective-upload experiment.

This runner deliberately performs *full adapter upload*.  It writes a Top-5
context compatible with the frozen V17 reader labeler, so reader evaluation is
an external second step rather than a training-time reward.  It never accepts
a final-test path.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from lora_blocks import adapter_state, inject_lora_blocks, load_adapter_state, state_bytes


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def title_id(dataset: str, title: str) -> str:
    normalized = " ".join(title.lower().split())
    return f"{dataset}:{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:20]}"


def support_titles(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", {})
    values = facts.get("title", []) if isinstance(facts, dict) else [value[0] for value in facts]
    return {" ".join(str(value).lower().split()) for value in values}


def documents(row: dict[str, Any], dataset: str) -> list[dict[str, str]]:
    context = row.get("context", {})
    if not isinstance(context, dict):
        return []
    answer, seen = [], set()
    for title, sentences in zip(context.get("title", []), context.get("sentences", [])):
        normalized = " ".join(str(title).lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        answer.append({"doc_id": title_id(dataset, str(title)), "title": str(title),
                       "text": " ".join(str(item).strip() for item in sentences if str(item).strip())})
    return answer


def pooled(model: nn.Module, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6), dim=-1)


def encode(model: nn.Module, tokenizer: Any, texts: list[str], device: torch.device, batch_size: int) -> torch.Tensor:
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            result.append(pooled(model, encoded).cpu())
    return torch.cat(result) if result else torch.empty((0, 768))


def make_model(model_name: str, device: torch.device, rank: int, alpha: float) -> nn.Module:
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    inject_lora_blocks(model, rank=rank, alpha=alpha)
    return model.to(device)


def query_prefix(question: str) -> str:
    return "Represent this sentence for searching relevant passages: " + question


def build_pairs(train_path: Path, assignment_path: Path, dataset: str, limit: int, clients: int, seed: int) -> dict[int, list[tuple[str, str]]]:
    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in rows(assignment_path)}
    output: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for row in rows(train_path):
        supports = support_titles(row)
        positive = next((doc for doc in documents(row, dataset) if " ".join(doc["title"].lower().split()) in supports), None)
        if positive is None:
            continue
        owner = assignment.get(positive["doc_id"])
        if owner is None:
            continue
        output[owner].append((query_prefix(str(row["question"])), positive["title"] + ": " + positive["text"]))
        if sum(map(len, output.values())) >= limit:
            break
    for values in output.values():
        random.Random(seed).shuffle(values)
    if not output or any(client < 0 or client >= clients for client in output):
        raise RuntimeError("failed to create valid client-local training pairs")
    return output


def contrastive_step(model: nn.Module, tokenizer: Any, batch: list[tuple[str, str]], device: torch.device) -> torch.Tensor:
    queries, passages = zip(*batch)
    q = tokenizer(list(queries), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    d = tokenizer(list(passages), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    query_embeddings, doc_embeddings = pooled(model, q), pooled(model, d)
    logits = query_embeddings @ doc_embeddings.T * 20.0
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=device))


def train_state(model: nn.Module, tokenizer: Any, initial: dict[str, torch.Tensor], pairs: list[tuple[str, str]], device: torch.device,
                steps: int, batch_size: int, lr: float, proximal: dict[str, torch.Tensor] | None = None,
                mu: float = 0.0, global_control: dict[str, torch.Tensor] | None = None,
                client_control: dict[str, torch.Tensor] | None = None) -> tuple[dict[str, torch.Tensor], float]:
    load_adapter_state(model, initial, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        batch = [pairs[(step * batch_size + offset) % len(pairs)] for offset in range(batch_size)]
        loss = contrastive_step(model, tokenizer, batch, device)
        if proximal is not None and mu > 0:
            current = dict(model.named_parameters())
            loss = loss + 0.5 * mu * sum((current[name] - value.to(device)).pow(2).sum() for name, value in proximal.items())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if global_control is not None and client_control is not None:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    parameter.grad.add_(global_control[name].to(device) - client_control[name].to(device))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return adapter_state(model), float(np.mean(losses))


def average_states(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    return {name: sum(state[name] * (weight / total) for state, weight in zip(states, weights)) for name in states[0]}


def train_method(args: argparse.Namespace, pairs: dict[int, list[tuple[str, str]]], tokenizer: Any, device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    global_model = make_model(args.model, device, args.rank, args.alpha)
    global_state = adapter_state(global_model)
    logs: list[dict[str, Any]] = []
    if args.method == "frozen":
        return global_state, logs
    if args.method == "centralized":
        joined = [pair for client_pairs in pairs.values() for pair in client_pairs]
        state, loss = train_state(global_model, tokenizer, global_state, joined, device, args.rounds * args.local_steps, args.batch_size, args.lr)
        logs.append({"round": 0, "method": args.method, "loss": loss, "participants": 1, "uploaded_bytes": 0})
        return state, logs
    controls = {client: {name: torch.zeros_like(value) for name, value in global_state.items()} for client in pairs}
    global_control = {name: torch.zeros_like(value) for name, value in global_state.items()}
    for round_id in range(args.rounds):
        rng = random.Random(args.seed + round_id)
        selected = sorted(rng.sample(sorted(pairs), min(args.participation, len(pairs))))
        local_states, weights, round_losses = [], [], []
        for client in selected:
            state, loss = train_state(global_model, tokenizer, global_state, pairs[client], device, args.local_steps, args.batch_size, args.lr,
                                      proximal=global_state if args.method == "fedprox" else None, mu=args.prox_mu if args.method == "fedprox" else 0.0,
                                      global_control=global_control if args.method == "scaffold" else None,
                                      client_control=controls[client] if args.method == "scaffold" else None)
            if args.method == "scaffold":
                denominator = max(args.local_steps * args.lr, 1e-8)
                controls[client] = {name: controls[client][name] - global_control[name] + (global_state[name] - state[name]) / denominator for name in state}
            local_states.append(state)
            weights.append(float(len(pairs[client])))
            round_losses.append(loss)
        if args.method == "local_only":
            global_state = local_states[0]
        else:
            global_state = average_states(local_states, weights)
        if args.method == "scaffold":
            global_control = {name: sum(controls[client][name] for client in selected) / len(selected) for name in global_control}
        logs.append({"round": round_id, "method": args.method, "loss": float(np.mean(round_losses)), "participants": len(selected),
                     "uploaded_bytes": state_bytes(global_state) * len(selected), "downloaded_bytes": state_bytes(global_state) * len(selected)})
    return global_state, logs


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.5] * len(values) if hi - lo < 1e-12 else [(value - lo) / (hi - lo) for value in values]


def evaluate_pool(model: nn.Module, tokenizer: Any, state: dict[str, torch.Tensor], source_path: Path, pool_path: Path, dataset: str,
                  output_pool: Path, output_contexts: Path, max_queries: int, device: torch.device, batch_size: int) -> dict[str, float]:
    load_adapter_state(model, state, device)
    source = {str(row["query_id"]): row for row in rows(source_path)}
    output_pool.parent.mkdir(parents=True, exist_ok=True)
    coverage, full_coverage, total = 0.0, 0.0, 0
    with output_pool.open("w", encoding="utf-8") as pool_handle, output_contexts.open("w", encoding="utf-8") as context_handle:
        for index, pool in enumerate(rows(pool_path)):
            if index >= max_queries:
                break
            query_id = str(pool["query_id"])
            source_row = source[query_id]
            candidates = list(pool["pool"])
            q = encode(model, tokenizer, [query_prefix(str(source_row["question"]))], device, batch_size)[0]
            d = encode(model, tokenizer, [doc["title"] + ": " + doc["text"] for doc in candidates], device, batch_size)
            dense = (d @ q).tolist()
            sparse = minmax([float(doc.get("sparse_score", 0.0)) for doc in candidates])
            for doc, dense_score, sparse_score in zip(candidates, dense, sparse):
                doc["dense_score"] = float(dense_score)
                doc["hybrid_score"] = 0.55 * float(dense_score) + 0.45 * float(sparse_score)
                doc["retrieval_score"] = doc["hybrid_score"]
            candidates.sort(key=lambda doc: (-float(doc["hybrid_score"]), str(doc["doc_id"])))
            top_five = [str(doc["doc_id"]) for doc in candidates[:5]]
            selected_titles = {" ".join(str(doc["title"]).lower().split()) for doc in candidates[:5]}
            gold = support_titles(source_row)
            coverage += len(selected_titles & gold) / max(1, len(gold))
            full_coverage += float(gold <= selected_titles)
            total += 1
            pool["pool"] = candidates
            pool["baseline_doc_ids"] = top_five
            pool["v19_adapter_rescored"] = True
            pool_handle.write(json.dumps(pool, ensure_ascii=False) + "\n")
            context_handle.write(json.dumps({"query_id": query_id, "trajectory_id": f"v19_{query_id}", "depth": 0,
                "candidate_type": "v19_retriever_top5", "context_doc_ids": top_five, "cheap_score": 0.0, "is_baseline": True}, ensure_ascii=False) + "\n")
    return {"queries": total, "support_recall_at_5": coverage / max(1, total), "complete_support_at_5": full_coverage / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "musique", "2wikimultihopqa"), required=True)
    parser.add_argument("--method", choices=("frozen", "centralized", "local_only", "fedavg", "fedprox", "scaffold"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-pool", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--max-development", type=int, default=100)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--participation", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "final" in str(args.train).lower() or "final" in str(args.development).lower() or "final" in str(args.development_pool).lower():
        raise ValueError("Stage 0 rejects final-test artifacts")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    pairs = build_pairs(args.train, args.assignment, args.dataset, args.train_limit, args.clients, args.seed)
    state, logs = train_method(args, pairs, tokenizer, device)
    model = make_model(args.model, device, args.rank, args.alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_pool(model, tokenizer, state, args.development, args.development_pool, args.dataset,
                            args.output_dir / "adapted_pool.jsonl", args.output_dir / "contexts.jsonl", args.max_development, device, args.embed_batch_size)
    torch.save({"adapter_state": state, "method": args.method, "seed": args.seed, "dataset": args.dataset}, args.output_dir / "adapter.pt")
    with (args.output_dir / "round_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "method", "loss", "participants", "uploaded_bytes", "downloaded_bytes"])
        writer.writeheader(); writer.writerows(logs)
    summary = {"status": "complete", "stage": "stage0_smoke", "method": args.method, "dataset": args.dataset,
               "seed": args.seed, "train_pairs": sum(map(len, pairs.values())), "adapter_bytes": state_bytes(state),
               "reader_evaluation": "pending_frozen_v17_reader", **metrics}
    (args.output_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
