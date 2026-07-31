"""Stable LoRA block injection for V19 retriever-only federated adaptation.

Each target linear layer owns one low-rank residual.  A block is a stable group
of target modules in a transformer layer, not an arbitrary tensor slice.  This
is deliberately small enough to make later byte-matched selection meaningful.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    layer: int
    module_type: str
    targets: tuple[str, ...]


class LoRALinear(nn.Module):
    """Frozen linear map plus a trainable, initially-zero low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = torch.nn.functional.linear(torch.nn.functional.linear(values, self.lora_a), self.lora_b)
        return self.base(values) + self.scale * residual


def default_block_specs(layer_count: int = 12) -> list[BlockSpec]:
    """Return attention/FFN blocks in six layers plus one pooling block."""
    if layer_count < 6:
        raise ValueError("V19 expects a BERT-style encoder with at least six layers")
    specs = []
    for layer in range(layer_count - 6, layer_count):
        specs.append(BlockSpec(
            block_id=f"layer_{layer:02d}_attention",
            layer=layer,
            module_type="attention_output",
            targets=(f"encoder.layer.{layer}.attention.output.dense",),
        ))
        specs.append(BlockSpec(
            block_id=f"layer_{layer:02d}_ffn",
            layer=layer,
            module_type="ffn_output",
            targets=(f"encoder.layer.{layer}.output.dense",),
        ))
    specs.append(BlockSpec("pooler", layer=layer_count, module_type="pooler", targets=("pooler.dense",)))
    return specs


def _resolve_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    parent: nn.Module = root
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    return parent, pieces[-1]


def inject_lora_blocks(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> list[BlockSpec]:
    """Inject the fixed schema and return it. Calling twice is an error."""
    # V19 adapts only transmitted LoRA residuals.  Freezing the whole base
    # encoder before replacement prevents an accidental full-model upload.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer_count = len(model.encoder.layer)
    specs = default_block_specs(layer_count)
    for spec in specs:
        for target in spec.targets:
            parent, name = _resolve_parent(model, target)
            module = getattr(parent, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{target} is not an nn.Linear; got {type(module)!r}")
            setattr(parent, name, LoRALinear(module, rank=rank, alpha=alpha))
    return specs


def block_parameters(model: nn.Module, specs: Iterable[BlockSpec]) -> OrderedDict[str, list[nn.Parameter]]:
    """Map every stable block ID to its trainable LoRA parameters."""
    output: OrderedDict[str, list[nn.Parameter]] = OrderedDict()
    for spec in specs:
        parameters: list[nn.Parameter] = []
        for target in spec.targets:
            parent, name = _resolve_parent(model, target)
            module = getattr(parent, name)
            if not isinstance(module, LoRALinear):
                raise TypeError(f"missing injected LoRA target {target}")
            parameters.extend((module.lora_a, module.lora_b))
        output[spec.block_id] = parameters
    return output


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detach only trainable adapter tensors, keeping base BGE immutable."""
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    current = dict(model.named_parameters())
    if set(current) & set(state) != set(state):
        missing = set(state) - set(current)
        raise KeyError(f"adapter state contains unknown tensors: {sorted(missing)}")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=device, dtype=current[name].dtype))


def state_bytes(state: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())
