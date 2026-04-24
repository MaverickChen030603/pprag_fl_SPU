from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence


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


class UploadSelector:
    """Strategy dispatcher for RP comparison methods."""

    def __init__(
        self,
        strategy: str,
        block_names: Sequence[str],
        topk: int,
        always_upload: Sequence[str] | None = None,
        seed: int = 0,
    ) -> None:
        self.strategy = strategy
        self.block_names = list(block_names)
        self.topk = int(topk)
        self.always_upload = list(always_upload or [])
        self.rng = random.Random(seed)

    def select(
        self,
        client_id: int,
        current_round: int,
        last_stats: Mapping[str, Mapping[str, float]] | None = None,
        hypernet_scores: Mapping[str, float] | None = None,
    ) -> SelectionResult:
        del client_id, current_round
        if self.strategy in {"full", "fedavg_full"}:
            return SelectionResult("full", ["__ALL__"], {block: 1.0 for block in self.block_names})
        if self.strategy == "random":
            blocks = random_blocks(self.block_names, self.topk, self.rng, self.always_upload)
            return SelectionResult("random", blocks, {block: 1.0 if block in blocks else 0.0 for block in self.block_names})
        if self.strategy == "static_top":
            blocks = static_top_layers(self.block_names, self.topk, self.always_upload)
            return SelectionResult("static_top", blocks, {block: 1.0 if block in blocks else 0.0 for block in self.block_names})
        if self.strategy == "delta_norm":
            blocks = delta_norm_blocks(self.block_names, self.topk, last_stats, self.always_upload)
            scores = {block: float((last_stats or {}).get(block, {}).get("l2", 0.0)) for block in self.block_names}
            return SelectionResult("delta_norm", blocks, scores)
        if self.strategy == "hypernet":
            scores = dict(hypernet_scores or {})
            if not scores:
                blocks = static_top_layers(self.block_names, self.topk, self.always_upload)
                scores = {block: 1.0 if block in blocks else 0.0 for block in self.block_names}
            else:
                blocks = select_topk(_rank_by_score(scores), self.topk, self.always_upload)
            return SelectionResult("hypernet", blocks, scores)
        raise ValueError(f"Unknown upload selection strategy: {self.strategy}")

