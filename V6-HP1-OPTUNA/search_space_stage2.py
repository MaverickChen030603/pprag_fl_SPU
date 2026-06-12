from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Stage2TrialConfig:
    topk: int
    warmup: int
    score_mode: str
    budget_mode: str
    hard_query_scale: float
    hard_client_threshold: float
    adaptive_expand_threshold: float
    utility_expand_threshold: float
    adaptive_shrink_threshold: float
    history_window: int
    use_hard_query_weighting: bool
    use_utility_memory: bool
    layerwise_budget: bool
    payload_penalty: float


def suggest_stage2_trial_config(trial: Any, *, payload_penalty_mode: str, fixed_payload_penalty: float) -> Stage2TrialConfig:
    """Narrow second-pass search around the best low-payload V6-HP1-OPTUNA region."""

    if payload_penalty_mode == "search":
        payload_penalty = trial.suggest_categorical("payload_penalty", [0.10, 0.25, 0.40])
    else:
        payload_penalty = fixed_payload_penalty

    return Stage2TrialConfig(
        topk=2,
        warmup=0,
        score_mode=trial.suggest_categorical("score_mode", ["value", "downstream_value"]),
        budget_mode=trial.suggest_categorical("budget_mode", ["fixed", "adaptive_v6"]),
        hard_query_scale=trial.suggest_float("hard_query_scale", 0.70, 1.20, step=0.05),
        hard_client_threshold=trial.suggest_float("hard_client_threshold", 0.68, 0.84, step=0.02),
        adaptive_expand_threshold=trial.suggest_float("adaptive_expand_threshold", 0.72, 0.86, step=0.02),
        utility_expand_threshold=trial.suggest_float("utility_expand_threshold", 1.15, 1.60, step=0.05),
        adaptive_shrink_threshold=trial.suggest_float("adaptive_shrink_threshold", 0.40, 0.56, step=0.02),
        history_window=trial.suggest_categorical("history_window", [3, 5, 7]),
        use_hard_query_weighting=trial.suggest_categorical("use_hard_query_weighting", [True, False]),
        use_utility_memory=False,
        layerwise_budget=trial.suggest_categorical("layerwise_budget", [True, False]),
        payload_penalty=float(payload_penalty),
    )
