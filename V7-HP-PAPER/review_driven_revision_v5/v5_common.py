#!/usr/bin/env python3
"""Shared, dependency-light utilities for the V5 review-driven revision."""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
V4 = PAPER_ROOT / "opportunity_aware_semantic_generation_v4"
V4_COMPLETION = PAPER_ROOT / "v4_submission_completion"
V4_FINAL = PAPER_ROOT / "v4_final_submission_refinement"
CONFIG_PATH = HERE / "configs" / "revision_v5.json"


def read_json(path: os.PathLike[str] | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: os.PathLike[str] | str) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_name = handle.name
    os.replace(temp_name, path)


def write_json(path: os.PathLike[str] | str, value: Any) -> None:
    _atomic_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: os.PathLike[str] | str, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    _atomic_text(Path(path), text)


def write_text(path: os.PathLike[str] | str, text: str) -> None:
    _atomic_text(Path(path), text.rstrip() + "\n")


def config() -> dict[str, Any]:
    value = read_json(CONFIG_PATH)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid or missing V5 config: {CONFIG_PATH}")
    return value


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def answer_scores(prediction: str, gold: str) -> dict[str, float]:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    em = float(pred == target)
    pred_tokens, target_tokens = pred.split(), target.split()
    common = Counter(pred_tokens) & Counter(target_tokens)
    overlap = sum(common.values())
    if not pred_tokens or not target_tokens:
        f1 = float(pred_tokens == target_tokens)
    elif overlap == 0:
        f1 = 0.0
    else:
        precision = overlap / len(pred_tokens)
        recall = overlap / len(target_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return {"answer_em": em, "answer_f1": f1}


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap(
    deltas: Sequence[float], *, seed: int = 20260714, samples: int = 5000
) -> dict[str, float]:
    values = [float(value) for value in deltas]
    if not values:
        return {"n": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean([values[rng.randrange(len(values))] for _ in values]))
    estimate = mean(values)
    same_or_more_extreme = sum(
        1 for item in estimates if (item <= 0.0 if estimate >= 0 else item >= 0.0)
    )
    return {
        "n": len(values),
        "delta": estimate,
        "ci_low": percentile(estimates, 0.025),
        "ci_high": percentile(estimates, 0.975),
        "p_value": min(1.0, 2.0 * same_or_more_extreme / samples),
    }


def artifact(path: Path, *, minimum_rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.exists() and minimum_rows is not None:
        if path.suffix == ".jsonl":
            count = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
        else:
            value = read_json(path, [])
            count = len(value) if isinstance(value, list) else 1
        result.update({"rows": count, "minimum_rows": minimum_rows, "complete": count >= minimum_rows})
    return result


def marker(value: Any, fallback: str = "[NEEDS MEASUREMENT]") -> Any:
    return fallback if value is None else value

