from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_bert_block_name(param_name: str) -> str:
    """Map a BERT/BGE parameter name to a communication block name."""
    if param_name.startswith("model."):
        param_name = param_name[len("model."):]
    parts = param_name.split(".")
    if not parts:
        return param_name
    if parts[0] == "embeddings":
        return "embeddings"
    if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layer":
        return ".".join(parts[:3])
    if parts[0] == "pooler":
        return "pooler"
    return parts[0]


def build_block_map(
    state_dict: Mapping[str, torch.Tensor],
    block_strategy: str = "bert",
) -> Dict[str, List[str]]:
    """Build block -> parameter names mapping for selective upload."""
    if block_strategy not in {"bert", "top_level"}:
        raise ValueError(f"Unsupported block_strategy: {block_strategy}")

    block_map: Dict[str, List[str]] = OrderedDict()
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            continue
        if block_strategy == "bert":
            block_name = infer_bert_block_name(name)
        else:
            block_name = name.split(".")[0]
        block_map.setdefault(block_name, []).append(name)
    return block_map


def clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device | str = "cpu",
    detach: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    cloned = OrderedDict()
    for name, tensor in state_dict.items():
        value = tensor.detach() if detach else tensor
        cloned[name] = value.clone().to(device)
    return cloned


def subtract_state_dict(
    new_state: Mapping[str, torch.Tensor],
    old_state: Mapping[str, torch.Tensor],
    device: torch.device | str = "cpu",
) -> OrderedDict[str, torch.Tensor]:
    delta = OrderedDict()
    for name, new_tensor in new_state.items():
        if name not in old_state or not torch.is_tensor(new_tensor):
            continue
        if not torch.is_floating_point(new_tensor):
            continue
        old_tensor = old_state[name].to(new_tensor.device)
        delta[name] = (new_tensor.detach() - old_tensor.detach()).to(device)
    return delta


def block_delta_stats(
    delta: Mapping[str, torch.Tensor],
    block_map: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = OrderedDict()
    for block, names in block_map.items():
        sq_norm = 0.0
        numel = 0
        max_abs = 0.0
        for name in names:
            if name not in delta:
                continue
            tensor = delta[name].detach().float().cpu()
            sq_norm += float(torch.sum(tensor * tensor).item())
            numel += int(tensor.numel())
            if tensor.numel() > 0:
                max_abs = max(max_abs, float(torch.max(torch.abs(tensor)).item()))
        l2_norm = sq_norm ** 0.5
        mean_l2 = l2_norm / max(numel, 1)
        stats[block] = {
            "l2": l2_norm,
            "mean_l2": mean_l2,
            "max_abs": max_abs,
            "numel": float(numel),
        }
    return stats


def filter_delta_by_blocks(
    delta: Mapping[str, torch.Tensor],
    block_map: Mapping[str, Sequence[str]],
    upload_blocks: Iterable[str],
) -> OrderedDict[str, torch.Tensor]:
    upload_blocks = set(upload_blocks)
    if "__ALL__" in upload_blocks:
        return OrderedDict((name, value.cpu()) for name, value in delta.items())

    selected_names = set()
    for block in upload_blocks:
        selected_names.update(block_map.get(block, []))
    return OrderedDict(
        (name, value.cpu()) for name, value in delta.items() if name in selected_names
    )


def estimate_payload_ratio(
    block_map: Mapping[str, Sequence[str]],
    state_dict: Mapping[str, torch.Tensor],
    upload_blocks: Iterable[str],
) -> float:
    upload_blocks = set(upload_blocks)
    total = 0
    selected = 0
    for block, names in block_map.items():
        for name in names:
            if name not in state_dict:
                continue
            tensor = state_dict[name]
            if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
                continue
            total += tensor.numel()
            if "__ALL__" in upload_blocks or block in upload_blocks:
                selected += tensor.numel()
    return selected / max(total, 1)


class BlockImportanceHyperNetwork(nn.Module):
    """Small hypernetwork that predicts upload importance per client and block."""

    def __init__(
        self,
        client_num: int,
        block_names: Sequence[str],
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        stat_dim: int = 3,
    ) -> None:
        super().__init__()
        self.client_num = client_num
        self.block_names = list(block_names)
        self.block_to_id = {name: idx for idx, name in enumerate(self.block_names)}
        self.client_embedding = nn.Embedding(client_num, embedding_dim)
        self.block_embedding = nn.Embedding(len(self.block_names), embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        client_id: int,
        block_names: Sequence[str],
        stat_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stat_features is None:
            stat_features = torch.zeros(len(block_names), 3, device=self.device)
        else:
            stat_features = stat_features.to(self.device).float()

        client_ids = torch.full(
            (len(block_names),),
            int(client_id),
            dtype=torch.long,
            device=self.device,
        )
        block_ids = torch.tensor(
            [self.block_to_id[name] for name in block_names],
            dtype=torch.long,
            device=self.device,
        )
        features = torch.cat(
            [
                self.client_embedding(client_ids),
                self.block_embedding(block_ids),
                stat_features,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.mlp(features)).squeeze(-1)

    def score_blocks(
        self,
        client_id: int,
        block_names: Sequence[str],
        stat_features: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            scores = self.forward(client_id, block_names, stat_features)
        return {
            block: float(score.detach().cpu().item())
            for block, score in zip(block_names, scores)
        }


def stats_to_features(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]] | None,
) -> torch.Tensor:
    if not stats:
        return torch.zeros(len(block_names), 3)

    l2_values = torch.tensor(
        [float(stats.get(block, {}).get("l2", 0.0)) for block in block_names],
        dtype=torch.float,
    )
    mean_l2_values = torch.tensor(
        [float(stats.get(block, {}).get("mean_l2", 0.0)) for block in block_names],
        dtype=torch.float,
    )
    max_abs_values = torch.tensor(
        [float(stats.get(block, {}).get("max_abs", 0.0)) for block in block_names],
        dtype=torch.float,
    )

    def normalize(values: torch.Tensor) -> torch.Tensor:
        denom = values.max().clamp_min(1e-12)
        return values / denom

    return torch.stack(
        [normalize(l2_values), normalize(mean_l2_values), normalize(max_abs_values)],
        dim=-1,
    )


def target_from_stats(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
) -> torch.Tensor:
    targets = torch.tensor(
        [float(stats.get(block, {}).get("l2", 0.0)) for block in block_names],
        dtype=torch.float,
    )
    return targets / targets.max().clamp_min(1e-12)


def select_topk_blocks(
    scores: Mapping[str, float],
    topk: int,
    min_score: float = 0.0,
    always_upload: Sequence[str] | None = None,
) -> List[str]:
    if topk <= 0 or topk >= len(scores):
        return ["__ALL__"]
    always_upload = list(always_upload or [])
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = []
    for block, score in ranked:
        if score < min_score and len(selected) >= topk:
            break
        selected.append(block)
        if len(selected) >= topk:
            break
    for block in always_upload:
        if block in scores and block not in selected:
            selected.append(block)
    return selected


def supervised_hn_update(
    hypernet: BlockImportanceHyperNetwork,
    optimizer: torch.optim.Optimizer,
    client_id: int,
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
) -> float:
    features = stats_to_features(block_names, stats).to(hypernet.device)
    targets = target_from_stats(block_names, stats).to(hypernet.device)
    predictions = hypernet(client_id, block_names, features)
    loss = F.mse_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())

