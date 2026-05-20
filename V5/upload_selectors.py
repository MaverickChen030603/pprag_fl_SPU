from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

from budget_allocator import (
    allocate_client_budget,
    allocate_client_budget_v5,
    allocate_layerwise_budget,
    compute_value_density,
)


def _rank_by_score(scores: Mapping[str, float]) -> List[str]:
    return [block for block, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def select_topk(
    ranked_blocks: Sequence[str],
    topk: int,
    always_upload: Sequence[str] | None = None,
) -> List[str]:
    ranked_blocks = list(ranked_blocks)
    if topk <= 0 or topk >= len(ranked_blocks):
        return ["__ALL__"]
    selected = list(ranked_blocks[:topk])
    for block in always_upload or []:
        if block in ranked_blocks and block not in selected:
            selected.append(block)
    return selected


def random_blocks(
    block_names: Sequence[str],
    topk: int,
    rng: random.Random,
    always_upload: Sequence[str] | None = None,
) -> List[str]:
    block_names = list(block_names)
    if topk <= 0 or topk >= len(block_names):
        return ["__ALL__"]
    selected = rng.sample(block_names, topk)
    for block in always_upload or []:
        if block in block_names and block not in selected:
            selected.append(block)
    return selected


def static_top_layers(
    block_names: Sequence[str],
    topk: int,
    always_upload: Sequence[str] | None = None,
) -> List[str]:
    def layer_key(block: str) -> tuple[int, int, str]:
        if block == "pooler":
            return (3, 10_000, block)
        match = re.match(r"encoder\.layer\.(\d+)$", block)
        if match:
            return (2, int(match.group(1)), block)
        if block == "embeddings":
            return (0, -1, block)
        return (1, -1, block)

    ranked = sorted(block_names, key=layer_key, reverse=True)
    return select_topk(ranked, topk, always_upload)


def delta_norm_blocks(
    block_names: Sequence[str],
    topk: int,
    stats: Mapping[str, Mapping[str, float]] | None,
    always_upload: Sequence[str] | None = None,
) -> List[str]:
    if not stats:
        return static_top_layers(block_names, topk, always_upload)
    scores = {block: float(stats.get(block, {}).get("l2", 0.0)) for block in block_names}
    return select_topk(_rank_by_score(scores), topk, always_upload)


@dataclass(frozen=True)
class SelectionResult:
    strategy: str
    upload_blocks: List[str]
    scores: Dict[str, float]
    budget_topk: int
    predicted_budget_ratio: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


class UploadSelector:
    def __init__(
        self,
        strategy: str,
        block_names: Sequence[str],
        topk: int,
        always_upload: Sequence[str] | None = None,
        seed: int = 0,
        budget_mode: str = "fixed",
        adaptive_min_topk: int = 1,
        adaptive_max_topk: int = 7,
        adaptive_scale: float = 1.0,
        layerwise_budget: bool = False,
    ) -> None:
        self.strategy = strategy
        self.block_names = list(block_names)
        self.topk = int(topk)
        self.always_upload = list(always_upload or [])
        self.rng = random.Random(seed)
        self.budget_mode = budget_mode
        self.adaptive_min_topk = adaptive_min_topk
        self.adaptive_max_topk = adaptive_max_topk
        self.adaptive_scale = adaptive_scale
        self.layerwise_budget = layerwise_budget

    def _budget_topk(self, predicted_budget_ratio: float | None) -> int:
        if self.budget_mode not in {"adaptive", "adaptive_v5"}:
            return self.topk
        return allocate_client_budget(
            base_topk=self.topk,
            predicted_budget_ratio=predicted_budget_ratio,
            min_topk=self.adaptive_min_topk,
            max_topk=self.adaptive_max_topk,
            adaptive_scale=self.adaptive_scale,
        )

    def _apply_layerwise_budget(self, ranked_blocks: Sequence[str], budget_topk: int) -> List[str]:
        if not self.layerwise_budget:
            return list(ranked_blocks[:budget_topk])
        return allocate_layerwise_budget(list(ranked_blocks), budget_topk)

    def select(
        self,
        client_id: int,
        current_round: int,
        last_stats: Mapping[str, Mapping[str, float]] | None = None,
        hypernet_scores: Mapping[str, float] | None = None,
        block_costs: Mapping[str, int] | None = None,
        predicted_budget_ratio: float | None = None,
        score_mode: str = "importance",
        client_hardness: float = 0.0,
        hard_client_threshold: float = 0.55,
        hard_client_bonus_topk: int = 1,
        utility_ratio: float = 1.0,
        adaptive_expand_threshold: float = 0.62,
        adaptive_shrink_threshold: float = 0.42,
        utility_expand_threshold: float = 1.15,
        hard_budget_only: bool = True,
    ) -> SelectionResult:
        del client_id, current_round
        if self.strategy in {"full", "fedavg_full"}:
            return SelectionResult("full", ["__ALL__"], {block: 1.0 for block in self.block_names}, budget_topk=len(self.block_names))
        if self.strategy == "random":
            blocks = random_blocks(self.block_names, self.topk, self.rng, self.always_upload)
            return SelectionResult(
                "random",
                blocks,
                {block: 1.0 if block in blocks else 0.0 for block in self.block_names},
                budget_topk=self.topk,
            )
        if self.strategy == "static_top":
            blocks = static_top_layers(self.block_names, self.topk, self.always_upload)
            return SelectionResult(
                "static_top",
                blocks,
                {block: 1.0 if block in blocks else 0.0 for block in self.block_names},
                budget_topk=self.topk,
            )
        if self.strategy == "delta_norm":
            blocks = delta_norm_blocks(self.block_names, self.topk, last_stats, self.always_upload)
            scores = {block: float((last_stats or {}).get(block, {}).get("l2", 0.0)) for block in self.block_names}
            return SelectionResult("delta_norm", blocks, scores, budget_topk=self.topk)

        scores = dict(hypernet_scores or {})
        if not scores:
            blocks = static_top_layers(self.block_names, self.topk, self.always_upload)
            return SelectionResult(
                self.strategy,
                blocks,
                {block: 1.0 if block in blocks else 0.0 for block in self.block_names},
                budget_topk=self.topk,
            )

        if self.strategy == "hypernet_v2":
            ranked = _rank_by_score(scores)
            blocks = select_topk(ranked, self.topk, self.always_upload)
            return SelectionResult("hypernet_v2", blocks, scores, budget_topk=self.topk)

        if self.budget_mode == "adaptive_v5":
            budget_topk = allocate_client_budget_v5(
                base_topk=self.topk,
                predicted_budget_ratio=predicted_budget_ratio,
                min_topk=self.adaptive_min_topk,
                max_topk=self.adaptive_max_topk,
                adaptive_scale=self.adaptive_scale,
                client_hardness=client_hardness,
                hard_client_threshold=hard_client_threshold,
                hard_client_bonus_topk=hard_client_bonus_topk,
                utility_ratio=utility_ratio,
                adaptive_expand_threshold=adaptive_expand_threshold,
                adaptive_shrink_threshold=adaptive_shrink_threshold,
                utility_expand_threshold=utility_expand_threshold,
                hard_budget_only=hard_budget_only,
            )
        else:
            budget_topk = self._budget_topk(predicted_budget_ratio)
        ranking_scores = scores
        if score_mode in {"value", "downstream_value"}:
            ranking_scores = compute_value_density(scores, block_costs)
        ranked = _rank_by_score(ranking_scores)
        if budget_topk <= 0 or budget_topk >= len(self.block_names):
            blocks = ["__ALL__"]
        else:
            chosen = self._apply_layerwise_budget(ranked, budget_topk)
            blocks = list(chosen[:budget_topk])
            for block in self.always_upload:
                if block in self.block_names and block not in blocks:
                    blocks.append(block)
        return SelectionResult(
            "hypernet_v5" if self.strategy == "hypernet_v5" else "hypernet_v3",
            blocks,
            ranking_scores,
            budget_topk=budget_topk,
            predicted_budget_ratio=float(predicted_budget_ratio or 0.0),
            metadata={
                "score_mode": 1.0 if score_mode == "value" else 0.0,
                "utility_ratio": float(utility_ratio),
                "client_hardness": float(client_hardness),
            },
        )
