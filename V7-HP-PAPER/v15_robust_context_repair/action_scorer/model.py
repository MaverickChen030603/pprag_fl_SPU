from __future__ import annotations

import torch
from torch import nn


class DirectMultiReaderScorer(nn.Module):
    """Predict three deltas and two harm logits for every frozen reader."""

    def __init__(self, input_dim: int, readers: int, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.readers = readers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, readers * 5),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).reshape(-1, self.readers, 5)


def multitask_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    query_groups: list[str],
    regression_weight: float = 1.0,
    harm_weight: float = 0.5,
    ranking_weight: float = 0.5,
    margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = nn.functional.smooth_l1_loss(predictions[:, :, :3], labels[:, :, :3])
    harm = nn.functional.binary_cross_entropy_with_logits(predictions[:, :, 3:], labels[:, :, 3:])
    predicted_joint = predictions[:, :, 2].mean(dim=1)
    actual_joint = labels[:, :, 2].mean(dim=1)
    pair_losses = []
    for query_id in sorted(set(query_groups)):
        indices = [index for index, value in enumerate(query_groups) if value == query_id]
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                difference = actual_joint[left] - actual_joint[right]
                if abs(float(difference.detach())) < 1e-5:
                    continue
                sign = torch.sign(difference)
                pair_losses.append(torch.relu(margin - sign * (predicted_joint[left] - predicted_joint[right])))
    ranking = torch.stack(pair_losses).mean() if pair_losses else predictions.sum() * 0.0
    total = regression_weight * regression + harm_weight * harm + ranking_weight * ranking
    return total, {"regression": float(regression.detach()), "harm": float(harm.detach()), "ranking": float(ranking.detach()), "total": float(total.detach())}

