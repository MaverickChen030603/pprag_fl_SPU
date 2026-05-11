from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_bert_block_name(param_name: str) -> str:
    if param_name.startswith("model."):
        param_name = param_name[len("model.") :]
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
    if block_strategy not in {"bert", "top_level"}:
        raise ValueError(f"Unsupported block_strategy: {block_strategy}")
    block_map: Dict[str, List[str]] = OrderedDict()
    for name, value in state_dict.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            continue
        block = infer_bert_block_name(name) if block_strategy == "bert" else name.split(".")[0]
        block_map.setdefault(block, []).append(name)
    return block_map


def clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device | str = "cpu",
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, tensor.detach().clone().to(device)) for name, tensor in state_dict.items())


def subtract_state_dict(
    new_state: Mapping[str, torch.Tensor],
    old_state: Mapping[str, torch.Tensor],
    device: torch.device | str = "cpu",
) -> OrderedDict[str, torch.Tensor]:
    delta = OrderedDict()
    for name, new_tensor in new_state.items():
        if name not in old_state or not torch.is_tensor(new_tensor) or not torch.is_floating_point(new_tensor):
            continue
        delta[name] = (new_tensor.detach() - old_state[name].to(new_tensor.device).detach()).to(device)
    return delta


def block_delta_stats(
    delta: Mapping[str, torch.Tensor],
    block_map: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = OrderedDict()
    for block, names in block_map.items():
        sq_norm = 0.0
        abs_sum = 0.0
        max_abs = 0.0
        numel = 0
        for name in names:
            if name not in delta:
                continue
            tensor = delta[name].detach().float().cpu()
            sq_norm += float(torch.sum(tensor * tensor).item())
            abs_sum += float(torch.sum(torch.abs(tensor)).item())
            numel += int(tensor.numel())
            if tensor.numel() > 0:
                max_abs = max(max_abs, float(torch.max(torch.abs(tensor)).item()))
        l2 = sq_norm ** 0.5
        stats[block] = {
            "l2": l2,
            "mean_abs": abs_sum / max(numel, 1),
            "mean_l2": l2 / max(numel, 1),
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
    return OrderedDict((name, value.cpu()) for name, value in delta.items() if name in selected_names)


def block_parameter_counts(
    block_map: Mapping[str, Sequence[str]],
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, int]:
    counts = OrderedDict()
    for block, names in block_map.items():
        total = 0
        for name in names:
            value = state_dict.get(name)
            if torch.is_tensor(value) and torch.is_floating_point(value):
                total += int(value.numel())
        counts[block] = total
    return counts


def estimate_payload_ratio(
    block_map: Mapping[str, Sequence[str]],
    state_dict: Mapping[str, torch.Tensor],
    upload_blocks: Iterable[str],
) -> float:
    counts = block_parameter_counts(block_map, state_dict)
    total = sum(counts.values())
    upload_blocks = set(upload_blocks)
    if "__ALL__" in upload_blocks:
        selected = total
    else:
        selected = sum(count for block, count in counts.items() if block in upload_blocks)
    return selected / max(total, 1)


def normalize_vector(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    return values / values.max().clamp_min(1e-12)


def stats_to_features(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]] | None,
) -> torch.Tensor:
    if not stats:
        return torch.zeros(len(block_names), 4)
    l2 = torch.tensor([float(stats.get(block, {}).get("l2", 0.0)) for block in block_names], dtype=torch.float32)
    mean_abs = torch.tensor([float(stats.get(block, {}).get("mean_abs", 0.0)) for block in block_names], dtype=torch.float32)
    max_abs = torch.tensor([float(stats.get(block, {}).get("max_abs", 0.0)) for block in block_names], dtype=torch.float32)
    numel = torch.tensor([float(stats.get(block, {}).get("numel", 0.0)) for block in block_names], dtype=torch.float32)
    return torch.stack(
        [
            normalize_vector(l2),
            normalize_vector(mean_abs),
            normalize_vector(max_abs),
            normalize_vector(numel),
        ],
        dim=-1,
    )


def target_from_stats(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
    score_mode: str = "importance",
) -> torch.Tensor:
    if score_mode == "value":
        target = torch.tensor(
            [
                float(stats.get(block, {}).get("mean_l2", 0.0))
                for block in block_names
            ],
            dtype=torch.float32,
        )
    else:
        target = torch.tensor(
            [float(stats.get(block, {}).get("l2", 0.0)) for block in block_names],
            dtype=torch.float32,
        )
    return normalize_vector(target)


class ValueAwareHyperNetwork(nn.Module):
    def __init__(
        self,
        client_num: int,
        block_names: Sequence[str],
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        current_stat_dim: int = 4,
        history_dim: int = 6,
        use_client_embedding: bool = True,
        use_history_features: bool = True,
        use_block_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.client_num = client_num
        self.block_names = list(block_names)
        self.block_to_id = {block: idx for idx, block in enumerate(self.block_names)}
        self.use_client_embedding = use_client_embedding
        self.use_history_features = use_history_features
        self.use_block_embedding = use_block_embedding
        self.current_stat_dim = current_stat_dim
        self.history_dim = history_dim if use_history_features else 0
        self.embedding_dim = embedding_dim

        if use_client_embedding:
            self.client_embedding = nn.Embedding(client_num, embedding_dim)
        else:
            self.register_parameter("client_embedding", None)
        if use_block_embedding:
            self.block_embedding = nn.Embedding(len(self.block_names), embedding_dim)
        else:
            self.register_parameter("block_embedding", None)

        input_dim = current_stat_dim + self.history_dim
        if use_client_embedding:
            input_dim += embedding_dim
        if use_block_embedding:
            input_dim += embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.budget_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_features(
        self,
        client_id: int,
        block_names: Sequence[str],
        current_features: torch.Tensor | None = None,
        history_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block_names = list(block_names)
        if current_features is None:
            current_features = torch.zeros(len(block_names), self.current_stat_dim, device=self.device)
        else:
            current_features = current_features.to(self.device).float()
        parts = [current_features]
        if self.use_history_features:
            if history_features is None:
                history_features = torch.zeros(len(block_names), self.history_dim, device=self.device)
            else:
                history_features = history_features.to(self.device).float()
            parts.append(history_features)
        if self.use_client_embedding:
            client_ids = torch.full((len(block_names),), int(client_id), dtype=torch.long, device=self.device)
            parts.append(self.client_embedding(client_ids))
        if self.use_block_embedding:
            block_ids = torch.tensor([self.block_to_id[block] for block in block_names], dtype=torch.long, device=self.device)
            parts.append(self.block_embedding(block_ids))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        client_id: int,
        block_names: Sequence[str],
        current_features: torch.Tensor | None = None,
        history_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self._build_features(client_id, block_names, current_features, history_features)
        return torch.sigmoid(self.mlp(features)).squeeze(-1)

    def score_blocks(
        self,
        client_id: int,
        block_names: Sequence[str],
        current_features: torch.Tensor | None = None,
        history_features: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        with torch.no_grad():
            scores = self.forward(client_id, block_names, current_features, history_features)
        return {block: float(score.detach().cpu().item()) for block, score in zip(block_names, scores)}

    def predict_budget_ratio(
        self,
        client_id: int,
        block_names: Sequence[str],
        current_features: torch.Tensor | None = None,
        history_features: torch.Tensor | None = None,
    ) -> float:
        with torch.no_grad():
            features = self._build_features(client_id, block_names, current_features, history_features)
            pooled = torch.mean(features, dim=0, keepdim=True)
            ratio = self.budget_head(pooled).squeeze().detach().cpu().item()
        return float(ratio)


def supervised_hn_update(
    hypernet: ValueAwareHyperNetwork,
    optimizer: torch.optim.Optimizer,
    client_id: int,
    block_names: Sequence[str],
    current_stats: Mapping[str, Mapping[str, float]],
    history_features: torch.Tensor | None = None,
    score_mode: str = "importance",
) -> float:
    current_features = stats_to_features(block_names, current_stats).to(hypernet.device)
    targets = target_from_stats(block_names, current_stats, score_mode=score_mode).to(hypernet.device)
    predictions = hypernet(client_id, block_names, current_features, history_features)
    loss = F.mse_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
