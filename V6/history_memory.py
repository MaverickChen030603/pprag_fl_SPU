from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import torch


def _ema(old: float, new: float, momentum: float = 0.8) -> float:
    return momentum * old + (1.0 - momentum) * new


@dataclass
class BlockRecord:
    selected_count: int = 0
    last_selected_round: int = 0
    importance_ema: float = 0.0
    value_ema: float = 0.0
    selected_ema: float = 0.0
    payload_ema: float = 0.0
    downstream_utility_ema: float = 0.0
    hard_query_utility_ema: float = 0.0
    client_hardness_ema: float = 0.0


@dataclass
class SelectionMemory:
    history_window: int = 5
    momentum: float = 0.8
    client_state: Dict[int, Dict[str, BlockRecord]] = field(default_factory=lambda: defaultdict(dict))
    client_rounds: Dict[int, int] = field(default_factory=dict)

    def _record(self, client_id: int, block_name: str) -> BlockRecord:
        block_state = self.client_state.setdefault(client_id, {})
        if block_name not in block_state:
            block_state[block_name] = BlockRecord()
        return block_state[block_name]

    def update(
        self,
        client_id: int,
        block_names: Sequence[str],
        stats: Mapping[str, Mapping[str, float]],
        selected_blocks: Sequence[str],
        round_idx: int,
        payload_ratio: float,
        downstream_utility_map: Mapping[str, float] | None = None,
        hard_query_utility_map: Mapping[str, float] | None = None,
        client_hardness: float = 0.0,
    ) -> None:
        selected_set = set(selected_blocks)
        if "__ALL__" in selected_set:
            selected_set = set(block_names)
        self.client_rounds[client_id] = max(round_idx, self.client_rounds.get(client_id, 0))
        for block_name in block_names:
            record = self._record(client_id, block_name)
            block_stats = stats.get(block_name, {})
            importance = float(block_stats.get("l2", 0.0))
            value = float(block_stats.get("mean_l2", 0.0))
            selected = 1.0 if block_name in selected_set else 0.0
            record.importance_ema = _ema(record.importance_ema, importance, self.momentum)
            record.value_ema = _ema(record.value_ema, value, self.momentum)
            record.selected_ema = _ema(record.selected_ema, selected, self.momentum)
            record.payload_ema = _ema(record.payload_ema, payload_ratio, self.momentum)
            record.downstream_utility_ema = _ema(
                record.downstream_utility_ema,
                float((downstream_utility_map or {}).get(block_name, 0.0)),
                self.momentum,
            )
            record.hard_query_utility_ema = _ema(
                record.hard_query_utility_ema,
                float((hard_query_utility_map or {}).get(block_name, 0.0)),
                self.momentum,
            )
            record.client_hardness_ema = _ema(record.client_hardness_ema, client_hardness, self.momentum)
            if selected > 0.0:
                record.selected_count += 1
                record.last_selected_round = round_idx

    def get_block_history(self, client_id: int, block_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
        current_round = max(self.client_rounds.get(client_id, 0), 1)
        history: Dict[str, Dict[str, float]] = {}
        client_blocks = self.client_state.get(client_id, {})
        for block_name in block_names:
            record = client_blocks.get(block_name, BlockRecord())
            rounds_seen = max(current_round, 1)
            recency = 0.0
            if record.last_selected_round > 0:
                distance = current_round - record.last_selected_round
                recency = max(0.0, 1.0 - distance / max(self.history_window, 1))
            history[block_name] = {
                "selection_freq": record.selected_count / rounds_seen,
                "selection_recency": recency,
                "importance_ema": record.importance_ema,
                "value_ema": record.value_ema,
                "selected_ema": record.selected_ema,
                "payload_ema": record.payload_ema,
                "downstream_utility_ema": record.downstream_utility_ema,
                "hard_query_utility_ema": record.hard_query_utility_ema,
                "client_hardness_ema": record.client_hardness_ema,
            }
        return history


def history_to_features(
    block_names: Sequence[str],
    history: Mapping[str, Mapping[str, float]] | None,
) -> torch.Tensor:
    if not history:
        return torch.zeros(len(block_names), 9)
    feature_order = [
        "selection_freq",
        "selection_recency",
        "importance_ema",
        "value_ema",
        "selected_ema",
        "payload_ema",
        "downstream_utility_ema",
        "hard_query_utility_ema",
        "client_hardness_ema",
    ]
    rows = []
    for block_name in block_names:
        block_history = history.get(block_name, {})
        rows.append([float(block_history.get(name, 0.0)) for name in feature_order])
    tensor = torch.tensor(rows, dtype=torch.float32)
    max_vals = torch.amax(torch.abs(tensor), dim=0).clamp_min(1e-12)
    return tensor / max_vals
