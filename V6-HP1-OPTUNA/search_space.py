from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrialConfig:
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


def suggest_trial_config(trial: Any) -> TrialConfig:
    """Sample a compact V6-HP1 search space for the first Optuna pass."""

    budget_mode = trial.suggest_categorical("budget_mode", ["fixed", "adaptive_v6"])
    return TrialConfig(
        topk=trial.suggest_categorical("topk", [2, 3, 4]),
        warmup=trial.suggest_categorical("warmup", [0, 1]),
        score_mode=trial.suggest_categorical("score_mode", ["downstream_value", "value"]),
        budget_mode=budget_mode,
        hard_query_scale=trial.suggest_float("hard_query_scale", 0.75, 1.75),
        hard_client_threshold=trial.suggest_float("hard_client_threshold", 0.62, 0.82),
        adaptive_expand_threshold=trial.suggest_float("adaptive_expand_threshold", 0.72, 0.90),
        utility_expand_threshold=trial.suggest_float("utility_expand_threshold", 1.15, 1.65),
        adaptive_shrink_threshold=trial.suggest_float("adaptive_shrink_threshold", 0.40, 0.58),
        history_window=trial.suggest_categorical("history_window", [3, 5, 7]),
        use_hard_query_weighting=trial.suggest_categorical("use_hard_query_weighting", [True, False]),
        use_utility_memory=trial.suggest_categorical("use_utility_memory", [True, False]),
        layerwise_budget=trial.suggest_categorical("layerwise_budget", [False, True]),
    )

