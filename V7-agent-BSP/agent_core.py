from __future__ import annotations

import math

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence


HP1_EARLY_EVIDENCE_BLOCKS = {"embeddings", "encoder.layer.0", "encoder.layer.1", "encoder.layer.2", "encoder.layer.3"}
HP1_BRIDGE_BLOCKS = {"encoder.layer.8", "encoder.layer.9", "encoder.layer.10", "encoder.layer.11", "pooler"}


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _ema(old: float, new: float, rho: float) -> float:
    return float(rho) * float(old) + (1.0 - float(rho)) * float(new)


def _normalise(values: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    raw = {key: max(0.0, float(values.get(key, 0.0))) for key in keys}
    max_value = max(raw.values(), default=0.0)
    if max_value <= 1e-12:
        return {key: 0.0 for key in keys}
    return {key: value / max_value for key, value in raw.items()}


def _hp1_early_evidence_prior(block_name: str, enabled: bool = True) -> float:
    if not enabled:
        return 0.0
    if block_name in HP1_EARLY_EVIDENCE_BLOCKS:
        return 1.0
    if block_name in HP1_BRIDGE_BLOCKS:
        return 0.25
    return 0.0


@dataclass
class AgentBlockMemory:
    utility_ema: float = 0.0
    hard_query_ema: float = 0.0
    rarity_ema: float = 0.0
    selection_ema: float = 0.0
    instability_ema: float = 0.0
    reward_mean: float = 0.0
    reward_count: int = 0
    last_selected: float = 0.0
    last_round: int = 0


@dataclass
class AgentMemory:
    """Lightweight client-agent memory M_i^t for budget-aligned block selection.

    The memory intentionally acts as a low-pass filter over noisy downstream/RAG
    feedback. It tracks utility EMA, hard-query EMA, rarity EMA and a per-block
    instability EMA measuring cross-round mask flips.
    """

    block_names: Sequence[str]
    rho: float = 0.8
    instability_rho: float = 0.7
    records: Dict[str, AgentBlockMemory] = field(default_factory=dict)
    round_index: int = 0

    def __post_init__(self) -> None:
        self.block_names = list(self.block_names)
        for block in self.block_names:
            self.records.setdefault(block, AgentBlockMemory())

    def update(
        self,
        selected_blocks: Iterable[str],
        observed_rewards: Mapping[str, float] | None = None,
        hard_query_alignment: Mapping[str, float] | None = None,
        client_rarity_score: float = 0.0,
        round_index: int | None = None,
    ) -> None:
        if round_index is not None:
            self.round_index = int(round_index)
        else:
            self.round_index += 1
        selected = set(selected_blocks)
        if "__ALL__" in selected:
            selected = set(self.block_names)
        rewards = observed_rewards or {}
        hard = hard_query_alignment or {}
        rarity = _clamp01(client_rarity_score)
        for block in self.block_names:
            record = self.records.setdefault(block, AgentBlockMemory())
            is_selected = 1.0 if block in selected else 0.0
            mask_flip = abs(is_selected - record.last_selected)
            reward = float(rewards.get(block, 0.0))
            record.utility_ema = _ema(record.utility_ema, reward, self.rho)
            record.hard_query_ema = _ema(record.hard_query_ema, float(hard.get(block, 0.0)), self.rho)
            record.rarity_ema = _ema(record.rarity_ema, rarity if is_selected else 0.0, self.rho)
            record.selection_ema = _ema(record.selection_ema, is_selected, self.rho)
            record.instability_ema = _ema(record.instability_ema, mask_flip, self.instability_rho)
            if is_selected:
                record.reward_count += 1
                record.reward_mean += (reward - record.reward_mean) / max(record.reward_count, 1)
            record.last_selected = is_selected
            record.last_round = self.round_index

    def instability_penalty(self, block_name: str) -> float:
        return _clamp01(self.records.get(block_name, AgentBlockMemory()).instability_ema)

    def jitter_rate(self) -> float:
        if not self.block_names:
            return 0.0
        return sum(self.instability_penalty(block) for block in self.block_names) / len(self.block_names)

    def as_history(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for block, record in self.records.items():
            out[block] = {
                "utility_ema": record.utility_ema,
                "hard_query_ema": record.hard_query_ema,
                "rarity_ema": record.rarity_ema,
                "selection_ema": record.selection_ema,
                "instability_penalty": record.instability_ema,
                "reward_mean": record.reward_mean,
                "reward_count": float(record.reward_count),
            }
        return out


@dataclass(frozen=True)
class AgentReward:
    local_utility: float
    memory_utility: float
    hard_query_alignment: float
    early_evidence_alignment: float
    client_rarity_score: float
    diversity_bonus: float
    instability_penalty: float
    exploration_bonus: float = 0.0

    def score(self, weights: Mapping[str, float]) -> float:
        return (
            float(weights.get("local", 0.35)) * self.local_utility
            + float(weights.get("memory", 0.20)) * self.memory_utility
            + float(weights.get("hard_query", 0.20)) * self.hard_query_alignment
            + float(weights.get("early_evidence", 0.0)) * self.early_evidence_alignment
            + float(weights.get("rarity", 0.10)) * self.client_rarity_score
            + float(weights.get("diversity", 0.05)) * self.diversity_bonus
            + float(weights.get("exploration", 0.0)) * self.exploration_bonus
            - float(weights.get("instability", 0.10)) * self.instability_penalty
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "local_utility": self.local_utility,
            "memory_utility": self.memory_utility,
            "hard_query_alignment": self.hard_query_alignment,
            "early_evidence_alignment": self.early_evidence_alignment,
            "client_rarity_score": self.client_rarity_score,
            "diversity_bonus": self.diversity_bonus,
            "instability_penalty": self.instability_penalty,
            "exploration_bonus": self.exploration_bonus,
        }


def estimate_query_difficulty(query_embedding, hard_query_centroid, threshold: float = 0.7) -> float:
    if np is None or query_embedding is None or hard_query_centroid is None:
        return 0.5
    q = np.asarray(query_embedding, dtype=float)
    c = np.asarray(hard_query_centroid, dtype=float)
    if q.size == 0 or c.size == 0 or np.all(c == 0):
        return 0.5
    nq = np.linalg.norm(q)
    nc = np.linalg.norm(c)
    if nq == 0 or nc == 0:
        return 0.5
    return float(np.clip(np.dot(q, c) / (nq * nc), 0.0, 1.0))


def get_dynamic_early_slots(
    difficulty: float,
    top_k: int = 3,
    difficulty_threshold_low: float = 0.3,
    difficulty_threshold_high: float = 0.7,
) -> int:
    if difficulty < difficulty_threshold_low:
        return 0
    if difficulty < difficulty_threshold_high:
        return 1
    return min(2, max(top_k - 1, 0))


class AgentScorer:
    """Budget-aligned V7 scorer for agent_rule_v7 and agent_bandit_v7.

    The scorer changes only block ordering. Top-K/payload budget is owned by the
    caller and must be enforced after scoring.
    """

    MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
        "stability_focused": {
            "local": 0.30,
            "memory": 0.30,
            "hard_query": 0.12,
            "rarity": 0.08,
            "diversity": 0.05,
            "instability": 0.22,
            "exploration": 0.00,
        },
        "hard_query_focused": {
            "local": 0.25,
            "memory": 0.16,
            "hard_query": 0.29,
            "early_evidence": 0.10,
            "rarity": 0.10,
            "diversity": 0.04,
            "instability": 0.02,
            "exploration": 0.00,
        },
        "diversity_focused": {
            "local": 0.24,
            "memory": 0.18,
            "hard_query": 0.18,
            "rarity": 0.22,
            "diversity": 0.14,
            "instability": 0.10,
            "exploration": 0.00,
        },
    }

    def __init__(
        self,
        block_names: Sequence[str],
        strategy: str = "agent_rule_v7",
        strategy_mode: str = "stability_focused",
        bandit_c: float = 0.65,
        use_early_prior: bool = True,
        use_coverage_replace: bool = True,
        use_memory_ema: bool = True,
        early_coverage_weight: float = 0.0,
        use_dynamic_slots: bool = False,
        difficulty_threshold_low: float = 0.3,
        difficulty_threshold_high: float = 0.7,
        use_instability_penalty: bool = True,
    ) -> None:
        self.block_names = list(block_names)
        self.strategy = strategy
        self.strategy_mode = strategy_mode if strategy_mode in self.MODE_WEIGHTS else "stability_focused"
        self.bandit_c = float(bandit_c)
        self.use_early_prior = bool(use_early_prior)
        self.use_coverage_replace = bool(use_coverage_replace)
        self.use_memory_ema = bool(use_memory_ema)
        self.early_coverage_weight = float(early_coverage_weight)
        self.use_dynamic_slots = bool(use_dynamic_slots)
        self.difficulty_threshold_low = float(difficulty_threshold_low)
        self.difficulty_threshold_high = float(difficulty_threshold_high)
        self.use_instability_penalty = bool(use_instability_penalty)

    def _weights_for_state(self, client_rarity_score: float, client_hardness: float) -> Dict[str, float]:
        mode = self.strategy_mode
        if self.strategy == "agent_rule_v7":
            if client_hardness >= 0.65:
                mode = "hard_query_focused"
            elif client_rarity_score >= 0.55:
                mode = "diversity_focused"
        weights = dict(self.MODE_WEIGHTS.get(mode, self.MODE_WEIGHTS["stability_focused"]))
        if self.strategy == "agent_bandit_v7":
            weights["exploration"] = max(weights.get("exploration", 0.0), 0.16)
            weights["memory"] = max(weights.get("memory", 0.0), 0.24)
        return weights

    def _early_evidence_alignment(
        self,
        block_name: str,
        history_block: Mapping[str, float],
        client_hardness: float,
    ) -> float:
        prior = _hp1_early_evidence_prior(block_name, self.use_early_prior)
        historical = max(
            float(history_block.get("early_evidence_ema", 0.0)),
            float(history_block.get("hard_query_utility_ema", 0.0)) * prior,
        )
        if self.strategy == "agent_rule_v7" and self.strategy_mode == "hard_query_focused":
            # HP1 is a multihop QA diagnostic: hard clients need at least one
            # low-layer evidence block in the same Top-K budget instead of only
            # high-layer bridge blocks. The prior changes ordering only.
            return max(historical, prior * max(0.72, _clamp01(client_hardness)))
        return max(historical, prior * _clamp01(client_hardness))

    def score_blocks(
        self,
        local_scores: Mapping[str, float],
        history: Mapping[str, Mapping[str, float]] | None = None,
        hard_query_alignment: Mapping[str, float] | None = None,
        client_rarity_score: float = 0.0,
        client_hardness: float = 0.0,
        current_round: int = 1,
    ) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        history = history or {}
        local_norm = _normalise(local_scores, self.block_names)
        hard_norm = _normalise(hard_query_alignment or {}, self.block_names)
        selected_ema = {
            block: float(history.get(block, {}).get("selected_ema", history.get(block, {}).get("selection_freq", 0.0)))
            for block in self.block_names
        }
        scores: Dict[str, float] = {}
        components: Dict[str, Dict[str, float]] = {}
        weights = self._weights_for_state(client_rarity_score, client_hardness)
        total_pulls = 1.0 + sum(float(history.get(block, {}).get("reward_count", 0.0)) for block in self.block_names)
        for block in self.block_names:
            h = history.get(block, {})
            memory_utility = max(
                float(h.get("utility_ema", 0.0)),
                float(h.get("downstream_utility_ema", 0.0)),
                float(h.get("value_ema", 0.0)),
            ) if self.use_memory_ema else 0.0
            hard_alignment = max(float(h.get("hard_query_ema", 0.0)), float(h.get("hard_query_utility_ema", 0.0)), hard_norm[block])
            early_alignment = self._early_evidence_alignment(block, h, client_hardness)
            if self.strategy == "agent_rule_v7" and self.strategy_mode == "hard_query_focused":
                hard_alignment = max(hard_alignment, 0.45 * early_alignment)
            if self.strategy == "agent_bandit_v7" and self.early_coverage_weight > 0.0:
                hard_alignment = max(hard_alignment, self.early_coverage_weight * early_alignment)
            instability = max(float(h.get("instability_penalty", 0.0)), abs(local_norm[block] - selected_ema.get(block, 0.0)))
            if not self.use_instability_penalty:
                instability = 0.0
            diversity_bonus = 1.0 - _clamp01(selected_ema.get(block, 0.0))
            pulls = 1.0 + float(h.get("reward_count", 0.0))
            exploration = self.bandit_c * math.sqrt(max(math.log(max(current_round, total_pulls, 2.0)), 0.0) / pulls)
            reward = AgentReward(
                local_utility=_clamp01(local_norm[block]),
                memory_utility=_clamp01(memory_utility),
                hard_query_alignment=_clamp01(hard_alignment),
                early_evidence_alignment=_clamp01(early_alignment),
                client_rarity_score=_clamp01(client_rarity_score),
                diversity_bonus=_clamp01(diversity_bonus),
                instability_penalty=_clamp01(instability),
                exploration_bonus=_clamp01(exploration),
            )
            scores[block] = reward.score(weights)
            comp = reward.to_dict()
            comp.update({"score": scores[block], "mode": float({"stability_focused": 0, "hard_query_focused": 1, "diversity_focused": 2}[self.strategy_mode])})
            components[block] = comp
        return scores, components
