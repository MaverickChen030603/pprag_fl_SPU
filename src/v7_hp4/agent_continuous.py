from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import math

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch is available on the training host.
    torch = None
    nn = object


def clamp01(value: float) -> float:
    if math.isnan(float(value)) or math.isinf(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class BlockState:
    """Feature state consumed by the HP4 continuous upload policy."""

    local_utility: float = 0.0
    memory_utility: float = 0.0
    hard_query_alignment: float = 0.0
    client_rarity_score: float = 0.0
    bridge_entity_overlap: float = 0.0
    rare_token_overlap: float = 0.0
    bridge_entity_match_ratio: float = 0.0
    diversity_bonus: float = 0.0
    instability_penalty: float = 0.0

    def as_vector(self) -> list[float]:
        return [
            clamp01(self.local_utility),
            clamp01(self.memory_utility),
            clamp01(self.hard_query_alignment),
            clamp01(self.client_rarity_score),
            clamp01(self.bridge_entity_overlap),
            clamp01(self.rare_token_overlap),
            clamp01(self.bridge_entity_match_ratio),
            clamp01(self.diversity_bonus),
            clamp01(self.instability_penalty),
        ]


class ContinuousUploadPolicy(nn.Module if torch is not None else object):
    """Sigmoid policy head that emits soft upload weights w in [0, 1].

    HP1-HP3 used the agent score mainly to rank or select discrete blocks. HP4
    keeps all blocks in the aggregation path and lets this policy modulate their
    contribution continuously.
    """

    feature_names = (
        "local_utility",
        "memory_utility",
        "hard_query_alignment",
        "client_rarity_score",
        "bridge_entity_overlap",
        "rare_token_overlap",
        "bridge_entity_match_ratio",
        "diversity_bonus",
        "instability_penalty",
    )

    def __init__(self, input_dim: int | None = None, hidden_dim: int = 32, init_bias: float = 0.0) -> None:
        if torch is None:
            raise RuntimeError("ContinuousUploadPolicy requires torch")
        super().__init__()
        input_dim = len(self.feature_names) if input_dim is None else int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(self.net(features)).squeeze(-1)

    @classmethod
    def tensor_from_states(cls, states: Sequence[BlockState], device: str | None = None) -> "torch.Tensor":
        if torch is None:
            raise RuntimeError("ContinuousUploadPolicy requires torch")
        return torch.tensor([s.as_vector() for s in states], dtype=torch.float32, device=device)

    def weights_from_states(self, states: Sequence[BlockState], device: str | None = None) -> list[float]:
        if torch is None:
            raise RuntimeError("ContinuousUploadPolicy requires torch")
        with torch.no_grad():
            weights = self(self.tensor_from_states(states, device=device)).detach().cpu().tolist()
        return [clamp01(w) for w in weights]


def heuristic_soft_weight(state: BlockState) -> float:
    """Deterministic fallback used for data construction and CPU sanity tests."""

    logit = (
        1.60 * clamp01(state.memory_utility)
        + 1.35 * clamp01(state.hard_query_alignment)
        + 1.25 * clamp01(state.bridge_entity_overlap)
        + 1.10 * clamp01(state.rare_token_overlap)
        + 1.20 * clamp01(state.bridge_entity_match_ratio)
        + 0.75 * clamp01(state.client_rarity_score)
        + 0.45 * clamp01(state.local_utility)
        + 0.30 * clamp01(state.diversity_bonus)
        - 1.10 * clamp01(state.instability_penalty)
        - 2.10
    )
    return clamp01(1.0 / (1.0 + math.exp(-logit)))


def weighted_average_embeddings(
    embeddings: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
) -> list[float]:
    """Aggregate all block embeddings with continuous upload weights."""

    keys = list(embeddings)
    if not keys:
        return []
    dim = len(embeddings[keys[0]])
    total = [0.0] * dim
    denom = 0.0
    for key in keys:
        w = clamp01(weights.get(key, 0.0))
        denom += w
        vec = embeddings[key]
        for i in range(dim):
            total[i] += float(vec[i]) * w
    if denom <= 1e-12:
        denom = float(len(keys))
        total = [0.0] * dim
        for key in keys:
            vec = embeddings[key]
            for i in range(dim):
                total[i] += float(vec[i])
    return [v / denom for v in total]


def counterfactual_marginal_rewards(
    actual_reward: float,
    counterfactual_rewards: Mapping[str, float],
) -> dict[str, float]:
    """R(a_j) = R_actual - R_counter(j)."""

    return {block: float(actual_reward) - float(counter) for block, counter in counterfactual_rewards.items()}


def compute_counterfactual_rewards(
    block_ids: Iterable[str],
    reward_fn,
) -> dict[str, float]:
    """Evaluate a reward function after zeroing each block.

    `reward_fn` receives a dictionary of block weights. This helper is deliberately
    small so training code can replace it with batched retrieval+reader calls.
    """

    ids = list(block_ids)
    actual_weights = {block: 1.0 for block in ids}
    actual = float(reward_fn(actual_weights))
    rewards = {}
    for block in ids:
        cf = dict(actual_weights)
        cf[block] = 0.0
        rewards[block] = actual - float(reward_fn(cf))
    return rewards
