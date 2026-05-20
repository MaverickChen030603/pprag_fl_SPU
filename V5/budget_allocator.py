from __future__ import annotations

from typing import Dict, Mapping, Sequence


def normalize_costs(block_costs: Mapping[str, int]) -> Dict[str, float]:
    if not block_costs:
        return {}
    max_cost = max(float(cost) for cost in block_costs.values()) or 1.0
    return {block: float(cost) / max_cost for block, cost in block_costs.items()}


def compute_value_density(
    scores: Mapping[str, float],
    block_costs: Mapping[str, int] | Mapping[str, float] | None,
) -> Dict[str, float]:
    if not block_costs:
        return dict(scores)
    normalized = normalize_costs({key: int(value) for key, value in block_costs.items()})
    density = {}
    for block, score in scores.items():
        density[block] = float(score) / max(normalized.get(block, 1.0), 1e-6)
    return density


def allocate_client_budget(
    base_topk: int,
    predicted_budget_ratio: float | None,
    min_topk: int,
    max_topk: int,
    adaptive_scale: float = 1.0,
) -> int:
    if predicted_budget_ratio is None:
        return max(min_topk, min(base_topk, max_topk))
    centered = max(0.5, min(1.5, 0.5 + adaptive_scale * float(predicted_budget_ratio)))
    topk = int(round(base_topk * centered))
    return max(min_topk, min(topk, max_topk))


def allocate_client_budget_v5(
    base_topk: int,
    predicted_budget_ratio: float | None,
    min_topk: int,
    max_topk: int,
    adaptive_scale: float = 1.0,
    client_hardness: float = 0.0,
    hard_client_threshold: float = 0.55,
    hard_client_bonus_topk: int = 1,
    utility_ratio: float = 1.0,
    adaptive_expand_threshold: float = 0.62,
    adaptive_shrink_threshold: float = 0.42,
    utility_expand_threshold: float = 1.15,
    hard_budget_only: bool = True,
) -> int:
    base = max(min_topk, min(base_topk, max_topk))
    ratio = float(predicted_budget_ratio if predicted_budget_ratio is not None else 0.5)
    topk = base
    should_expand = (
        ratio >= adaptive_expand_threshold
        and utility_ratio >= utility_expand_threshold
        and client_hardness >= hard_client_threshold
    )
    if not hard_budget_only and ratio <= adaptive_shrink_threshold:
        topk = max(min_topk, base - 1)
    elif not hard_budget_only:
        topk = allocate_client_budget(
            base_topk=base_topk,
            predicted_budget_ratio=predicted_budget_ratio,
            min_topk=min_topk,
            max_topk=max_topk,
            adaptive_scale=adaptive_scale,
        )
    if should_expand:
        topk = max(topk, base + int(hard_client_bonus_topk))
    return max(min_topk, min(topk, max_topk))


def allocate_layerwise_budget(
    ranked_blocks: Sequence[str],
    budget_topk: int,
) -> list[str]:
    if budget_topk <= 0 or budget_topk >= len(ranked_blocks):
        return list(ranked_blocks)
    selected: list[str] = []
    layer_buckets = {
        "pooler": [],
        "embeddings": [],
        "encoder": [],
        "other": [],
    }
    for block in ranked_blocks:
        if block == "pooler":
            layer_buckets["pooler"].append(block)
        elif block == "embeddings":
            layer_buckets["embeddings"].append(block)
        elif block.startswith("encoder.layer."):
            layer_buckets["encoder"].append(block)
        else:
            layer_buckets["other"].append(block)
    bucket_order = ["pooler", "encoder", "embeddings", "other"]
    while len(selected) < budget_topk and any(layer_buckets.values()):
        for bucket in bucket_order:
            if layer_buckets[bucket] and len(selected) < budget_topk:
                selected.append(layer_buckets[bucket].pop(0))
    return selected
