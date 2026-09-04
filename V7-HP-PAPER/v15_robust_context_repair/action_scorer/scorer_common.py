from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORBIDDEN_FEATURE_FRAGMENTS = (
    "gold", "is_support", "supporting_", "answer_presence", "answer_text",
    "target_", "reader_outcome", "realized_delta", "official_",
)
TARGET_NAMES = ("answer_delta", "sp_delta", "joint_delta", "answer_drop", "joint_drop")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_feature_names(names: Iterable[str]) -> list[str]:
    values = sorted(map(str, names))
    blocked = [name for name in values if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)]
    if blocked:
        raise ValueError(f"label-derived or outcome-derived inference features are forbidden: {blocked}")
    return values


def vectorize(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows], dtype=np.float32)


def label_tensor(rows: list[dict[str, Any]], readers: list[str]) -> np.ndarray:
    values = []
    for row in rows:
        per_reader = row.get("reader_labels", {})
        values.append([[float(per_reader[reader][name]) for name in TARGET_NAMES] for reader in readers])
    return np.asarray(values, dtype=np.float32)


def batches_by_query(rows: list[dict[str, Any]], max_rows: int, rng: np.random.Generator) -> list[list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["query_id"]), []).append(index)
    groups = list(grouped.values())
    rng.shuffle(groups)
    batches, current = [], []
    for group in groups:
        if current and len(current) + len(group) > max_rows:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches

