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
    l2 = float(block_stats.get("l2", mean_l2))
    if mean_abs <= 0.0:
        return mean_l2
    contrast = min(max_abs / max(mean_abs, 1e-8), 5.0)
    sharpness = min(mean_l2 / max(mean_abs, 1e-8), 5.0)
    stability = min(l2 / max(mean_l2, 1e-8), 5.0) if mean_l2 > 0.0 else 0.0
    hardness_boost = 1.0 + hard_query_scale * max(contrast - 1.0, 0.0) * 0.08
    selectivity = 1.0 + max(sharpness - 1.0, 0.0) * 0.06 + max(stability - 1.0, 0.0) * 0.04
    return mean_l2 * hardness_boost * selectivity


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
    sorted_utils = sorted(utility_map.values(), reverse=True)
    top_utilities = sorted_utils[: min(3, len(sorted_utils))]
    if not top_utilities:
        return 0.0
    baseline = max(sum(sorted_utils) / max(len(sorted_utils), 1), 1e-8)
    return float((sum(top_utilities) / len(top_utilities)) / baseline)
