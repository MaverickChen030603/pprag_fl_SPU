from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("missing query ID")


def lexical_tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", str(value).lower()) if len(token) > 1]


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def hop_count_without_labels(row: dict[str, Any], dataset: str) -> int | str:
    if dataset == "musique":
        match = re.match(r"(\d+)hop__", query_id(row))
        return int(match.group(1)) if match else "unknown"
    value = row.get("hop_count", row.get("num_hops"))
    return int(value) if value is not None else "unknown"
