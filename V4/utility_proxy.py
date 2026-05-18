from __future__ import annotations

from typing import Dict, Mapping, Sequence


def estimate_block_downstream_utility(
    block_name: str,
    stats: Mapping[str, Mapping[str, float]],
    hard_query_scale: float = 1.0,
) -> float:
    block_stats = stats.get(block_name, {})
    mean_l2 = float(block_stats.get("mean_l2", 0.0))
    max_abs = float(block_stats.get("max_abs", 0.0))
    mean_abs = float(block_stats.get("mean_abs", 0.0))
    hardness_boost = 1.0
    if mean_abs > 0:
        hardness_boost += hard_query_scale * min(max_abs / max(mean_abs, 1e-8), 5.0) * 0.1
    return mean_l2 * hardness_boost


def estimate_block_utility_map(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
    hard_query_scale: float = 1.0,
) -> Dict[str, float]:
    return {
        block_name: estimate_block_downstream_utility(block_name, stats, hard_query_scale=hard_query_scale)
        for block_name in block_names
    }


def estimate_client_downstream_proxy(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
    hard_query_scale: float = 1.0,
) -> float:
    if not block_names:
        return 0.0
    utility_map = estimate_block_utility_map(block_names, stats, hard_query_scale=hard_query_scale)
    top_utilities = sorted(utility_map.values(), reverse=True)[: min(3, len(utility_map))]
    if not top_utilities:
        return 0.0
    return float(sum(top_utilities) / len(top_utilities))
