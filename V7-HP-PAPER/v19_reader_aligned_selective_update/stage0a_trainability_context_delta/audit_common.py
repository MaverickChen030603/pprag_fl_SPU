"""Shared no-cache utilities for V19 Stage 0A engineering audits."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

V19 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V19 / "model"))
from lora_blocks import LoRALinear, inject_lora_blocks, load_adapter_state


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def query_prefix(question: str) -> str:
    return "Represent this sentence for searching relevant passages: " + question


def support_titles(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", {})
    values = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts]
    return {" ".join(str(x).lower().split()) for x in values}


def model_and_tokenizer(model_name: str, state: dict[str, torch.Tensor], device: torch.device) -> tuple[nn.Module, Any]:
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    inject_lora_blocks(model, rank=8, alpha=16.0)
    model = model.to(device).eval()
    load_adapter_state(model, state, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
    return model, tokenizer


def encode(model: nn.Module, tokenizer: Any, texts: list[str], device: torch.device, batch_size: int = 32) -> torch.Tensor:
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            values.append(torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-6), dim=-1).cpu())
    return torch.cat(values) if values else torch.empty((0, 768))


def grouped_state(state: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    groups: dict[str, list[torch.Tensor]] = {}
    for name, tensor in state.items():
        if "encoder.layer." in name:
            layer = int(name.split("encoder.layer.", 1)[1].split(".", 1)[0])
            group = f"layer_{layer:02d}_{'attention' if '.attention.output.' in name else 'ffn'}"
        elif ".pooler." in name:
            group = "pooler"
        else:
            group = "unknown"
        groups.setdefault(group, []).append(tensor)
    return groups


def flatten(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().float().cpu().reshape(-1) for value in values])
