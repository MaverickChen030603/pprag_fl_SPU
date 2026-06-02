from __future__ import annotations

from typing import Mapping, Sequence


def estimate_retrieval_stability(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
) -> float:
    if not block_names:
        return 0.0
    values = [float(stats.get(block_name, {}).get("mean_l2", 0.0)) for block_name in block_names]
    total = sum(values)
    if total <= 0:
        return 0.0
    top_values = sorted(values, reverse=True)[: min(3, len(values))]
    return float(sum(top_values) / max(total, 1e-8))
