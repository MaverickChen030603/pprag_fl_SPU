#!/usr/bin/env python3
"""Rescore a frozen pool with a saved V19 adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

V19 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V19 / "stage0_full_upload"))
sys.path.insert(0, str(V19 / "model"))
from lora_blocks import load_adapter_state, state_bytes
from run_stage0_viability import evaluate_pool, make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = make_model(args.model, device, 8, 16.0)
    state = torch.load(args.checkpoint, map_location="cpu")["adapter_state"]
    load_adapter_state(model, state, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_pool(
        model,
        tokenizer,
        state,
        args.development,
        args.pool,
        "hotpotqa",
        args.output_dir / "adapted_pool.jsonl",
        args.output_dir / "contexts.jsonl",
        args.max_queries,
        device,
        32,
    )
    result = {
        "status": "complete",
        "label": args.label,
        "queries": args.max_queries,
        "adapter_bytes": state_bytes(state),
        **metrics,
    }
    (args.output_dir / "rescore_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
