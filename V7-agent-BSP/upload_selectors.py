from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

try:
    from .agent_core import AgentScorer, get_dynamic_early_slots
except ImportError:
    from agent_core import AgentScorer, get_dynamic_early_slots

from budget_allocator import (
    allocate_client_budget,
    allocate_client_budget_v6,
    allocate_client_budget_v5,
    allocate_layerwise_budget,
    compute_value_density,
)


def _rank_by_score(scores: Mapping[str, float]) -> List[str]:
    return [block for block, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


HP1_EARLY_EVIDENCE_BLOCKS = {"embeddings", "encoder.layer.0", "encoder.layer.1", "encoder.layer.2", "encoder.layer.3"}
HP1_BRIDGE_GUARD_BLOCKS = {"encoder.layer.7", "encoder.layer.8", "encoder.layer.9", "encoder.layer.10", "encoder.layer.11", "pooler"}
PM_AGENT_STRATEGIES = {"agent_pm_dynamic_full", "agent_pm_bandit_slot"}

BSP_REWARD_MODE = {
    "agent_bsp_bandit_strict": "strict",
    "agent_bsp_bandit_retrieval": "retrieval",
    "agent_bsp_bandit_reader": "reader",
    "agent_bsp_memory_bandit_strict": "strict",
    "agent_bsp_memory_bandit_retrieval": "retrieval",
    "agent_bsp_memory_bandit_reader": "reader",
    "agent_bsp_memory_bandit_no_failure_state": "retrieval",
    "agent_bsp_memory_bandit_no_rarity_state": "retrieval",
    "agent_bsp_memory_bandit_no_instability_state": "retrieval",
    "agent_bsp_memory_bandit_no_history_state": "retrieval",
}
BSP_AGENT_STRATEGIES = set(BSP_REWARD_MODE)
DYNAMIC_PLANNING_STRATEGIES = {"agent_dynamic_slot", "agent_dynamic_no_hardness", "agent_dynamic_no_rarity", "agent_dynamic_no_bridge_guard"}
FIXED_SLOT_STRATEGIES = {"agent_fixed_slot_0", "agent_fixed_slot_1", "agent_fixed_slot_2", "agent_fixed_slot_3"}
PM_COMPAT_STRATEGIES = PM_AGENT_STRATEGIES | BSP_AGENT_STRATEGIES | DYNAMIC_PLANNING_STRATEGIES | FIXED_SLOT_STRATEGIES | {
    "agent_pm_dynamic_no_memory",
    "agent_pm_dynamic_no_failure_memory",
    "agent_pm_dynamic_no_rarity_memory",
    "agent_pm_dynamic_no_instability_penalty",
    "agent_pm_dynamic_no_utility_ema",
}


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
        if self.budget_mode not in {"adaptive", "adaptive_v5", "adaptive_v6"}:
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
        history_map: Mapping[str, Mapping[str, float]] | None = None,
        hard_query_alignment: Mapping[str, float] | None = None,
        client_rarity_score: float = 0.0,
        agent_strategy_mode: str = "stability_focused",
        use_early_prior: bool = True,
        use_coverage_replace: bool = True,
        use_memory_ema: bool = True,
        early_coverage_weight: float = 0.0,
        use_dynamic_slots: bool = False,
        difficulty_threshold_low: float = 0.3,
        difficulty_threshold_high: float = 0.7,
        fixed_early_slots: int = -1,
        use_pm_failure_memory: bool = True,
        use_pm_rarity_memory: bool = True,
        use_pm_instability_penalty: bool = True,
        use_bridge_guard: bool = True,
        disable_dynamic_hardness: bool = False,
    ) -> SelectionResult:
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

        if self.budget_mode == "adaptive_v6":
            budget_topk = allocate_client_budget_v6(
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
        elif self.budget_mode == "adaptive_v5":
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

        if self.strategy in {"agent_rule_v7", "agent_bandit_v7"} or self.strategy in PM_COMPAT_STRATEGIES:
            scorer_strategy = "agent_bandit_v7" if self.strategy == "agent_pm_bandit_slot" or self.strategy in BSP_AGENT_STRATEGIES else "agent_rule_v7"
            scorer_mode = "planning_memory" if self.strategy in PM_COMPAT_STRATEGIES else agent_strategy_mode
            effective_hard_alignment = hard_query_alignment if use_pm_failure_memory else {}
            effective_rarity_score = client_rarity_score if use_pm_rarity_memory else 0.0
            scorer = AgentScorer(
                self.block_names,
                strategy=scorer_strategy,
                strategy_mode=scorer_mode,
                use_early_prior=use_early_prior,
                use_coverage_replace=use_coverage_replace,
                use_memory_ema=use_memory_ema,
                early_coverage_weight=early_coverage_weight,
                use_dynamic_slots=use_dynamic_slots,
                difficulty_threshold_low=difficulty_threshold_low,
                difficulty_threshold_high=difficulty_threshold_high,
                use_instability_penalty=use_pm_instability_penalty,
            )
            agent_scores, components = scorer.score_blocks(
                local_scores=ranking_scores,
                history=history_map,
                hard_query_alignment=effective_hard_alignment,
                client_rarity_score=effective_rarity_score,
                client_hardness=client_hardness,
                current_round=current_round,
            )
            ranked = _rank_by_score(agent_scores)
            replacement_reason = "none"
            slot_policy = "none"
            if (self.strategy in {"agent_rule_v7", "agent_bandit_v7"} or self.strategy in PM_COMPAT_STRATEGIES) and agent_strategy_mode == "hard_query_focused" and use_coverage_replace and use_early_prior and 0 < budget_topk < len(self.block_names):
                early_ranked = [block for block in ranked if block in HP1_EARLY_EVIDENCE_BLOCKS]
                preview = list(ranked[:budget_topk])
                if fixed_early_slots >= 0:
                    early_slots = max(0, min(int(fixed_early_slots), int(budget_topk)))
                    slot_policy = f"fixed_{early_slots}"
                elif self.strategy == "agent_fixed_slot_0":
                    early_slots, slot_policy = 0, "fixed_0"
                elif self.strategy == "agent_fixed_slot_1":
                    early_slots, slot_policy = 1, "fixed_1"
                elif self.strategy == "agent_fixed_slot_2":
                    early_slots, slot_policy = 2, "fixed_2"
                elif self.strategy == "agent_fixed_slot_3":
                    early_slots, slot_policy = min(3, int(budget_topk)), "fixed_3"
                elif use_dynamic_slots or self.strategy in PM_COMPAT_STRATEGIES:
                    difficulty = 0.5 if disable_dynamic_hardness or self.strategy == "agent_dynamic_no_hardness" else max(0.0, min(1.0, float(client_hardness or 0.5)))
                    if self.strategy == "agent_pm_bandit_slot":
                        early_slots = get_dynamic_early_slots(difficulty + 0.10 * effective_rarity_score, budget_topk, difficulty_threshold_low, difficulty_threshold_high)
                        slot_policy = "ucb_proxy_slot"
                    elif self.strategy in BSP_AGENT_STRATEGIES:
                        reward_mode = BSP_REWARD_MODE.get(self.strategy, "strict")
                        memory_state = self.strategy.startswith("agent_bsp_memory_bandit") and self.strategy != "agent_bsp_memory_bandit_no_history_state"
                        failure_state = memory_state and use_pm_failure_memory
                        rarity_state = memory_state and use_pm_rarity_memory
                        instability_state = memory_state and use_pm_instability_penalty
                        failure_signal = sum(c.get("hard_query_alignment", 0.0) for c in components.values()) / max(len(components), 1)
                        base_signal = difficulty + 0.12 * effective_rarity_score + 0.08 * failure_signal
                        if reward_mode == "retrieval":
                            base_signal += 0.10 * effective_rarity_score + 0.06 * failure_signal
                        elif reward_mode == "reader":
                            base_signal += 0.06 * failure_signal - 0.04 * (0.0 if instability_state else 0.15)
                        elif reward_mode == "strict":
                            base_signal += 0.04 * effective_rarity_score
                        if memory_state:
                            base_signal += 0.08 * float(failure_state) + 0.05 * float(rarity_state) - 0.04 * float(instability_state and client_hardness < 0.35)
                        early_slots = get_dynamic_early_slots(base_signal, budget_topk, difficulty_threshold_low, difficulty_threshold_high)
                        if reward_mode == "reader" and difficulty >= difficulty_threshold_high:
                            early_slots = min(int(budget_topk), max(early_slots, 2))
                        bridge_slot = 1 if use_bridge_guard and (difficulty >= 0.55 or reward_mode in {"retrieval", "reader"}) else 0
                        target_slot = 1 if reward_mode == "reader" and difficulty >= 0.45 else 0
                        exploration_level = "high" if not memory_state else ("mid" if difficulty >= 0.55 else "low")
                        slot_policy = f"bsp_{reward_mode}_e{early_slots}_b{bridge_slot}_t{target_slot}_{exploration_level}"
                    else:
                        early_slots = get_dynamic_early_slots(difficulty + 0.15 * effective_rarity_score, budget_topk, difficulty_threshold_low, difficulty_threshold_high)
                        slot_policy = "dynamic"
                else:
                    early_slots = 1
                    slot_policy = "fixed_1"
                current_early = [block for block in preview if block in HP1_EARLY_EVIDENCE_BLOCKS]
                if early_ranked and early_slots > len(current_early):
                    needed = early_slots - len(current_early)
                    result = list(preview)
                    used = set(result)
                    candidates = [block for block in early_ranked if block not in used]
                    while needed > 0 and candidates:
                        replacement = candidates.pop(0)
                        for idx in range(len(result) - 1, -1, -1):
                            if result[idx] not in HP1_EARLY_EVIDENCE_BLOCKS:
                                result[idx] = replacement
                                used.add(replacement)
                                replacement_reason = "early_slot_fill"
                                needed -= 1
                                break
                        else:
                            break
                    ranked = result + [block for block in ranked if block not in set(result)]
                if self.strategy in BSP_AGENT_STRATEGIES and use_bridge_guard and 0 < budget_topk < len(self.block_names):
                    result = list(ranked[:budget_topk])
                    used = set(result)
                    bridge_candidates = [block for block in ranked if block in HP1_BRIDGE_GUARD_BLOCKS and block not in used]
                    has_bridge = any(block in HP1_BRIDGE_GUARD_BLOCKS for block in result)
                    if bridge_candidates and not has_bridge and (client_hardness >= 0.55 or BSP_REWARD_MODE.get(self.strategy) in {"retrieval", "reader"}):
                        replacement = bridge_candidates[0]
                        for idx in range(len(result) - 1, -1, -1):
                            if result[idx] not in HP1_EARLY_EVIDENCE_BLOCKS:
                                result[idx] = replacement
                                replacement_reason = "bsp_bridge_guard"
                                break
                        ranked = result + [block for block in ranked if block not in set(result)]
            if budget_topk <= 0 or budget_topk >= len(self.block_names):
                blocks = ["__ALL__"]
            else:
                chosen = self._apply_layerwise_budget(ranked, budget_topk)
                blocks = list(chosen[:budget_topk])
                # Same-Budget note: always_upload is retained for compatibility.
                # Strict experiments should either count it in top-k or set it empty.
                for block in self.always_upload:
                    if block in self.block_names and block not in blocks:
                        blocks.append(block)
            avg_instability = sum(c.get("instability_penalty", 0.0) for c in components.values()) / max(len(components), 1)
            avg_hard = sum(c.get("hard_query_alignment", 0.0) for c in components.values()) / max(len(components), 1)
            return SelectionResult(
                self.strategy,
                blocks,
                agent_scores,
                budget_topk=budget_topk,
                predicted_budget_ratio=float(predicted_budget_ratio or 0.0),
                metadata={
                    "client_hardness": float(client_hardness),
                    "client_rarity_score": float(client_rarity_score),
                    "agent_instability_penalty_mean": float(avg_instability),
                    "hard_query_alignment_mean": float(avg_hard),
                    "same_budget_topk": float(budget_topk),
                    "early_slot_num": float(locals().get("early_slots", -1)),
                    "slot_policy": slot_policy,
                    "replacement_reason": replacement_reason,
                    "pm_failure_memory_enabled": float(bool(use_pm_failure_memory)),
                    "pm_rarity_memory_enabled": float(bool(use_pm_rarity_memory)),
                    "pm_instability_penalty_enabled": float(bool(use_pm_instability_penalty)),
                    "bridge_guard_enabled": float(bool(use_bridge_guard and self.strategy != "agent_dynamic_no_bridge_guard")),
                    "bsp_reward_mode": float({"strict": 0, "retrieval": 1, "reader": 2}.get(BSP_REWARD_MODE.get(self.strategy, "strict"), -1)) if self.strategy in BSP_AGENT_STRATEGIES else -1.0,
                    "bsp_memory_state_enabled": float(self.strategy.startswith("agent_bsp_memory_bandit") and self.strategy != "agent_bsp_memory_bandit_no_history_state"),
                    "bsp_failure_state_enabled": float(bool(use_pm_failure_memory)),
                    "bsp_rarity_state_enabled": float(bool(use_pm_rarity_memory)),
                    "bsp_instability_state_enabled": float(bool(use_pm_instability_penalty)),
                    "bsp_bandit_action": slot_policy,
                },
            )

        if self.strategy in {"agent_tail_v7hp1", "agent_memory_v7hp1", "agent_oracle_v7hp1"}:
            base_scores = dict(ranking_scores)
            ranked = _rank_by_score(base_scores)
            tail_blocks = [
                "embeddings",
                "encoder.layer.0",
                "encoder.layer.1",
                "encoder.layer.2",
                "encoder.layer.3",
                "encoder.layer.7",
                "encoder.layer.8",
                "encoder.layer.11",
                "pooler",
            ]
            available_tail = [block for block in tail_blocks if block in self.block_names]
            if self.strategy == "agent_oracle_v7hp1":
                budget_pattern = [6, 5, 3, 1, 1]
                priority = available_tail + [block for block in ranked if block not in available_tail]
            elif self.strategy == "agent_tail_v7hp1":
                budget_pattern = [5, 4, 3, 2, 1]
                offset = (int(client_id) + int(current_round)) % max(len(available_tail), 1)
                rotated_tail = available_tail[offset:] + available_tail[:offset]
                priority = rotated_tail + [block for block in ranked if block not in rotated_tail]
            else:
                budget_pattern = [4, 3, 3, 3, 2]
                offset = (int(client_id) + int(current_round)) % max(len(ranked), 1)
                priority = ranked[offset:] + ranked[:offset]
            budget_topk = budget_pattern[int(client_id) % len(budget_pattern)]
            if client_hardness >= hard_client_threshold and self.strategy != "agent_memory_v7hp1":
                budget_topk = min(max(budget_topk, self.topk + max(hard_client_bonus_topk, 1)), self.adaptive_max_topk)
            budget_topk = max(self.adaptive_min_topk, min(int(budget_topk), self.adaptive_max_topk, len(self.block_names)))
            chosen = list(priority[:budget_topk])
            for block in self.always_upload:
                if block in self.block_names and block not in chosen:
                    chosen.append(block)
            return SelectionResult(
                self.strategy,
                chosen,
                base_scores,
                budget_topk=budget_topk,
                predicted_budget_ratio=float(predicted_budget_ratio or 0.0),
                metadata={"client_hardness": float(client_hardness), "utility_ratio": float(utility_ratio), "h1_tail_action": 1.0},
            )
        if budget_topk <= 0 or budget_topk >= len(self.block_names):
            blocks = ["__ALL__"]
        else:
            chosen = self._apply_layerwise_budget(ranked, budget_topk)
            blocks = list(chosen[:budget_topk])
            for block in self.always_upload:
                if block in self.block_names and block not in blocks:
                    blocks.append(block)
        return SelectionResult(
            "hypernet_v6" if self.strategy == "hypernet_v6" else ("hypernet_v5" if self.strategy == "hypernet_v5" else "hypernet_v3"),
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
