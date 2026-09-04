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
    variance = float(block_stats.get("variance", 0.0))
    if mean_abs <= 0.0:
        return mean_l2
    contrast = min(max_abs / max(mean_abs, 1e-8), 5.0)
    sharpness = min(mean_l2 / max(mean_abs, 1e-8), 5.0)
    stability = min(l2 / max(mean_l2, 1e-8), 5.0) if mean_l2 > 0.0 else 0.0
    concentration = min(variance / max(mean_l2, 1e-8), 5.0) if mean_l2 > 0.0 else 0.0
    hardness_boost = 1.0 + hard_query_scale * max(contrast - 1.0, 0.0) * 0.10
    selectivity = 1.0 + max(sharpness - 1.0, 0.0) * 0.07 + max(stability - 1.0, 0.0) * 0.05
    concentration_gate = 1.0 + max(concentration - 0.5, 0.0) * 0.05
    return mean_l2 * hardness_boost * selectivity * concentration_gate


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
    tail = sorted_utils[-min(3, len(sorted_utils)) :]
    head_mean = sum(top_utilities) / len(top_utilities)
    tail_mean = sum(tail) / len(tail)
    separation = max(head_mean - tail_mean, 0.0) / max(baseline, 1e-8)
    return float((head_mean / baseline) * (1.0 + 0.15 * separation))


def _reader_block_prior(block_name: str) -> float:
    if block_name == "pooler":
        return 0.82
    if block_name == "embeddings":
        return 0.52
    if block_name.startswith("encoder.layer."):
        try:
            idx = int(block_name.rsplit(".", 1)[-1])
        except ValueError:
            return 0.50
        # Reader-side QA tends to be most sensitive to upper semantic layers,
        # while early layers still help lexical evidence matching.
        if idx >= 10:
            return 1.00
        if idx >= 7:
            return 0.86
        if idx >= 4:
            return 0.66
        return 0.56
    return 0.50


def estimate_block_reader_feedback(
    block_name: str,
    stats: Mapping[str, Mapping[str, float]],
    client_hardness: float = 0.0,
    reader_feedback_scale: float = 1.0,
) -> float:
    block_stats = stats.get(block_name, {})
    mean_l2 = float(block_stats.get("mean_l2", 0.0))
    mean_abs = float(block_stats.get("mean_abs", 0.0))
    max_abs = float(block_stats.get("max_abs", 0.0))
    variance = float(block_stats.get("variance", 0.0))
    if mean_l2 <= 0.0 and mean_abs <= 0.0:
        return 0.0
    concentration = min(variance / max(mean_l2, 1e-8), 5.0) if mean_l2 > 0.0 else 0.0
    evidence_sharpness = min(max_abs / max(mean_abs, 1e-8), 5.0) if mean_abs > 0.0 else 1.0
    prior = _reader_block_prior(block_name)
    hard_boost = 1.0 + max(0.0, min(float(client_hardness), 1.0)) * 0.20
    sharp_boost = 1.0 + max(evidence_sharpness - 1.0, 0.0) * 0.05
    concentration_boost = 1.0 + max(concentration - 0.5, 0.0) * 0.04
    base = mean_l2 if mean_l2 > 0.0 else mean_abs
    return float(base * prior * hard_boost * sharp_boost * concentration_boost * max(reader_feedback_scale, 0.0))


def estimate_reader_feedback_map(
    block_names: Sequence[str],
    stats: Mapping[str, Mapping[str, float]],
    client_hardness: float = 0.0,
    reader_feedback_scale: float = 1.0,
) -> Dict[str, float]:
    return {
        block_name: estimate_block_reader_feedback(
            block_name,
            stats,
            client_hardness=client_hardness,
            reader_feedback_scale=reader_feedback_scale,
        )
        for block_name in block_names
    }


def blend_block_utility_with_reader_feedback(
    downstream_utility_map: Mapping[str, float],
    reader_feedback_map: Mapping[str, float],
    reader_feedback_weight: float = 0.25,
) -> Dict[str, float]:
    weight = max(0.0, min(float(reader_feedback_weight), 1.0))
    keys = set(downstream_utility_map) | set(reader_feedback_map)
    if not keys:
        return {}
    max_down = max([float(downstream_utility_map.get(k, 0.0)) for k in keys] or [0.0])
    max_reader = max([float(reader_feedback_map.get(k, 0.0)) for k in keys] or [0.0])
    blended: Dict[str, float] = {}
    for key in keys:
        down = float(downstream_utility_map.get(key, 0.0))
        reader = float(reader_feedback_map.get(key, 0.0))
        if max_down > 1e-12 and max_reader > 1e-12:
            reader = reader / max_reader * max_down
        blended[key] = (1.0 - weight) * down + weight * reader
    return blended


def apply_reader_step_reward(
    reader_feedback_map: Mapping[str, float],
    positive_reward: float = 10.0,
    negative_reward: float = -5.0,
    top_fraction: float = 0.25,
) -> Dict[str, float]:
    """Convert smooth reader proxy into cliff-style reward.

    Top evidence blocks receive a large positive reward, bottom blocks receive
    a penalty, and the middle receives zero. This deliberately high-contrast
    shaping is used by HP3 to force different block choices.
    """
    if not reader_feedback_map:
        return {}
    items = sorted(((k, float(v)) for k, v in reader_feedback_map.items()), key=lambda kv: kv[1], reverse=True)
    n = len(items)
    top_n = max(1, int(round(n * max(0.01, min(top_fraction, 1.0)))))
    bottom_n = max(1, int(round(n * max(0.01, min(top_fraction, 1.0)))))
    out = {k: 0.0 for k, _ in items}
    for k, _ in items[:top_n]:
        out[k] = float(positive_reward)
    for k, _ in items[-bottom_n:]:
        out[k] = float(negative_reward)
    return out
