from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def best_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["query_id"])].append(row)
    return [max(values, key=lambda row: (float(row["predicted_utility"]), str(row["action_id"]))) for _, values in sorted(groups.items())]


def apply_threshold(rows: list[dict[str, Any]], utility_threshold: float, harm_threshold: float) -> list[dict[str, Any]]:
    output = []
    for row in best_actions(rows):
        eligible = float(row["predicted_utility"]) > utility_threshold and float(row["predicted_answer_harm"]) < harm_threshold
        output.append({**row, "selected": int(eligible), "decision": "repair" if eligible else "exact_fallback"})
    return output


def observed_risk(rows: list[dict[str, Any]]) -> dict[str, float]:
    selected = [row for row in rows if row["selected"]]
    return {
        "coverage": len(selected) / len(rows) if rows else 0.0,
        "selected": len(selected),
        "answer_drop_rate": float(np.mean([row["answer_drop"] for row in selected])) if selected else 0.0,
        "joint_drop_rate": float(np.mean([row["joint_drop"] for row in selected])) if selected else 0.0,
        "mean_answer_delta": float(np.mean([row["answer_delta"] for row in selected])) if selected else 0.0,
        "mean_joint_delta": float(np.mean([row["joint_delta"] for row in selected])) if selected else 0.0,
    }


def threshold_grid(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    best = best_actions(rows)
    utility = np.quantile([float(row["predicted_utility"]) for row in best], np.linspace(0.0, 0.95, 20))
    harm = np.quantile([float(row["predicted_answer_harm"]) for row in best], np.linspace(0.05, 1.0, 20))
    return sorted(set((float(u), float(h)) for u in utility for h in harm))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

