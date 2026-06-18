from __future__ import annotations

from typing import Mapping, Sequence


def estimate_block_hardness(
    block_name: str,
    stats: Mapping[str, Mapping[str, float]],
) -> float:
    block_stats = stats.get(block_name, {})
    mean_abs = float(block_stats.get("mean_abs", 0.0))
    max_abs = float(block_stats.get("max_abs", 0.0))
    if mean_abs <= 0:
        return 0.0
    return min(max_abs / max(mean_abs, 1e-8), 5.0) / 5.0


def estimate_client_hardness(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
) -> float:
    if not block_names:
        return 0.0
    hardness_values = [estimate_block_hardness(block_name, stats) for block_name in block_names]
    if not hardness_values:
        return 0.0
    return float(sum(hardness_values) / len(hardness_values))
