#!/usr/bin/env python3
"""Train the V15 direct multi-reader delta/harm scorer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from model import DirectMultiReaderScorer, multitask_loss
from scorer_common import batches_by_query, label_tensor, read_jsonl, validate_feature_names, vectorize


def evaluate_loss(model, features, labels, groups, device) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(features, device=device))
        loss, _ = multitask_loss(prediction, torch.as_tensor(labels, device=device), groups)
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows, dev_rows = read_jsonl(args.train), read_jsonl(args.development)
    feature_names = validate_feature_names(train_rows[0]["features"].keys())
    readers = sorted(train_rows[0]["reader_labels"].keys())
    train_x, dev_x = vectorize(train_rows, feature_names), vectorize(dev_rows, feature_names)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x, dev_x = (train_x - mean) / std, (dev_x - mean) / std
    train_y, dev_y = label_tensor(train_rows, readers), label_tensor(dev_rows, readers)
    train_groups = [str(row["query_id"]) for row in train_rows]
    dev_groups = [str(row["query_id"]) for row in dev_rows]

    device = torch.device(args.device)
    model = DirectMultiReaderScorer(len(feature_names), len(readers), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history, best, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for indices in batches_by_query(train_rows, args.batch_rows, rng):
            x = torch.as_tensor(train_x[indices], device=device)
            y = torch.as_tensor(train_y[indices], device=device)
            groups = [train_groups[index] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss, parts = multitask_loss(prediction, y, groups)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite policy loss at epoch {epoch}")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            optimizer.step()
            losses.append((parts["total"], gradient_norm))
        dev_loss = evaluate_loss(model, dev_x, dev_y, dev_groups, device)
        record = {"epoch": epoch, "train_loss": float(np.mean([value[0] for value in losses])), "dev_loss": dev_loss, "gradient_norm": float(np.mean([value[1] for value in losses]))}
        history.append(record)
        print(json.dumps(record), flush=True)
        if dev_loss < best - 1e-5:
            best, stale = dev_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feature_names": feature_names, "readers": readers, "mean": mean, "std": std, "hidden_dim": args.hidden_dim, "seed": args.seed, "best_dev_loss": best}, args.checkpoint)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps({"status": "complete", "best_dev_loss": best, "epochs": len(history), "history": history}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

